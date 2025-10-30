import argparse
import csv
import numpy as np
import librosa


def read_kotlin_csv(path):
    params = {}
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        # Read header params until we hit the rows= line
        for _ in range(8):
            key, value = next(reader)
            params[key] = value
        # Remaining lines are mel rows
        for line in reader:
            if not line:
                continue
            rows.append([float(x) for x in line])
    mel = np.array(rows, dtype=np.float32)
    return params, mel


def compute_librosa_mel(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        htk=False,
        norm="slaney",
    )
    # Disable clipping (top_db=None) to match Kotlin
    S_db = librosa.power_to_db(S, ref=np.max, top_db=None)
    return S_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to the same audio used on Android")
    ap.add_argument("csv", help="Path to mel_dump.csv exported from Android")
    ap.add_argument("--tolerance", type=float, default=1.5, help="Allowed absolute dB diff")
    args = ap.parse_args()

    params, mel_k = read_kotlin_csv(args.csv)
    mel_l = compute_librosa_mel(args.audio)

    # Ensure same shape by trimming to min dims
    h = min(mel_k.shape[0], mel_l.shape[0])
    w = min(mel_k.shape[1], mel_l.shape[1])
    mel_k = mel_k[:h, :w]
    mel_l = mel_l[:h, :w]
    # Trim 2 time columns from edges to avoid padding/centering boundary effects
    if w > 4:
        mel_k = mel_k[:, 2:-2]
        mel_l = mel_l[:, 2:-2]

    diff = np.abs(mel_k - mel_l)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    p95 = float(np.percentile(diff, 95))

    print("Params from CSV:", params)
    print(f"Shape aligned to: {h}x{w}")
    print(f"Max abs diff: {max_diff:.4f} dB")
    print(f"Mean abs diff: {mean_diff:.4f} dB")
    print(f"95th pct abs diff: {p95:.4f} dB")

    if max_diff <= args.tolerance:
        print("MATCH: within tolerance")
    else:
        print("MISMATCH: exceeds tolerance")


if __name__ == "__main__":
    main()


