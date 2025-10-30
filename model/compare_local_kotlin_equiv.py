import argparse
import numpy as np
import librosa


def hann_window(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)).astype(np.float32)


def stft_magnitude_squared(x: np.ndarray, n_fft: int, hop: int, center: bool = True) -> np.ndarray:
    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect")
    win = hann_window(n_fft)
    n_frames = 1 + max(0, (len(x) - n_fft) // hop)
    spec = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    frame_buf = np.zeros(n_fft, dtype=np.float32)
    start = 0
    for t in range(n_frames):
        frame = x[start:start + n_fft]
        frame_buf[:] = 0.0
        L = min(len(frame), n_fft)
        frame_buf[:L] = frame[:L] * win[:L]
        # rfft returns positive freqs only
        fft = np.fft.rfft(frame_buf.astype(np.float32), n=n_fft)
        spec[t, :] = (fft.real ** 2 + fft.imag ** 2).astype(np.float32)
        start += hop
    return spec


def hz_to_mel_slaney(hz: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.where(hz >= min_log_hz,
                   min_log_mel + np.log(hz / min_log_hz) / logstep,
                   hz / f_sp)
    return mel


def mel_to_hz_slaney(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.where(mel >= min_log_mel,
                  min_log_hz * np.exp((mel - min_log_mel) * logstep),
                  mel * f_sp)
    return hz


def mel_filterbank_slaney(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0
    m_min = hz_to_mel_slaney(np.array([f_min]))[0]
    m_max = hz_to_mel_slaney(np.array([f_max]))[0]
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz_slaney(m_points)
    # Map to fft bins
    bins = np.floor((n_fft + 1) * f_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        if f_m_minus < f_m:
            fb[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / max(1, (f_m - f_m_minus))
        if f_m < f_m_plus:
            fb[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / max(1, (f_m_plus - f_m))
    # Slaney normalization (equal area per filter)
    for m in range(n_mels):
        f_left = f_points[m]
        f_right = f_points[m + 2]
        enorm = 2.0 / (f_right - f_left) if f_right > f_left else 0.0
        fb[m, :] *= enorm
    return fb


def kotlin_equiv_mel_db(y, sr=22050, n_fft=2048, hop=512, n_mels=128):
    # keep our STFT (reflect padding + Hann), swap in librosa’s mel filters
    power_spec = stft_magnitude_squared(y.astype(np.float32), n_fft=n_fft, hop=hop, center=True)
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=False, norm="slaney")  # <- swap
    mel_power = (power_spec @ fb.T).astype(np.float32).T
    # small floor to avoid rare zero-power spikes
    mel_power = np.maximum(mel_power, 1e-10)
    ref = float(mel_power.max())
    mel_db = 10.0 * np.log10(mel_power / ref)
    return mel_db.astype(np.float32)


def librosa_mel_db(y, sr=22050, n_fft=2048, hop=512, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        center=True, htk=False, norm="slaney"
    )
    return librosa.power_to_db(S, ref=np.max, top_db=None).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Compare Kotlin-equivalent mel vs librosa on the same audio")
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--tolerance", type=float, default=1.5, help="Allowed absolute dB diff")
    args = ap.parse_args()

    y, _ = librosa.load(args.audio, sr=22050)
    mel_k = kotlin_equiv_mel_db(y)
    mel_l = librosa_mel_db(y)

    h = min(mel_k.shape[0], mel_l.shape[0])
    w = min(mel_k.shape[1], mel_l.shape[1])
    mel_k = mel_k[:h, :w]
    mel_l = mel_l[:h, :w]
    # Ignore 2 edge columns to remove padding boundary effects
    if w > 4:
        mel_k = mel_k[:, 2:-2]
        mel_l = mel_l[:, 2:-2]

    diff = np.abs(mel_k - mel_l)
    print(f"Shape: {h}x{w}")
    print(f"Max abs diff: {diff.max():.4f} dB")
    print(f"Mean abs diff: {diff.mean():.4f} dB")
    print(f"95th pct abs diff: {np.percentile(diff,95):.4f} dB")
    if diff.max() <= args.tolerance:
        print("MATCH: within tolerance")
    else:
        print("MISMATCH: exceeds tolerance")


if __name__ == "__main__":
    main()


