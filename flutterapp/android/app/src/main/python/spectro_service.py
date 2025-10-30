import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for saving figures
import matplotlib.pyplot as plt
import soundfile as sf
import os


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio same as first script."""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = np.mean(y, axis=1)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def mel_spectrogram_match_first(
    path: str,
    output_dir: str,
    label_prefix: str,
    index: int,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256
):
    """Generate mel spectrogram *identical* to first script, and save as .png."""
    try:
        y = _load_audio(path, target_sr=sr)
        if len(y) == 0:
            print(f"⚠️ Empty file: {path}")
            return

        # === Step 1: identical feature extraction ===
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # === Step 2: identical normalization ===
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)

        # === Step 3: identical plotting parameters ===
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(S_db, sr=sr, cmap='magma')  # same colormap
        plt.axis('off')
        plt.tight_layout(pad=0)

        # === Step 4: identical file name pattern ===
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"{label_prefix}_{index:04d}.png"
        output_path = os.path.join(output_dir, output_name)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
