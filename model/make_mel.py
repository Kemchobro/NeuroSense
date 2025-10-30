import argparse
import os
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_mel_db(audio_path: str, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        htk=False,           # Slaney scale
        norm="slaney",      # Slaney normalization
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def save_png(mel_db: np.ndarray, out_png: str, sr: int = 22050):
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mel_db, sr=sr, cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_csv(mel_db: np.ndarray, out_csv: str, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    lines = []
    lines.append("sr,22050\n")
    lines.append("n_fft,2048\n")
    lines.append("hop_length,512\n")
    lines.append("n_mels,128\n")
    lines.append("center,True\n")
    lines.append("mel_scale,slaney\n")
    lines.append("norm,slaney\n")
    lines.append("rows=mel_bands,cols=time_frames\n")
    for m in range(mel_db.shape[0]):
        row = ",".join(str(float(x)) for x in mel_db[m])
        lines.append(row + "\n")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.writelines(lines)


def process_file(path: str, out_dir: str, write_csv: bool):
    mel_db = compute_mel_db(path)
    base = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(out_dir, f"{base}_mel.png")
    save_png(mel_db, out_png)
    if write_csv:
        out_csv = os.path.join(out_dir, f"{base}_mel.csv")
        save_csv(mel_db, out_csv)
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Generate mel spectrogram PNGs (and optional CSV) using training params.")
    ap.add_argument("input", help="Path to a WAV/MP3/M4A file or a directory of audio files")
    ap.add_argument("--out", default="model_out", help="Output directory (default: model_out)")
    ap.add_argument("--csv", action="store_true", help="Also write a CSV in the Android export format")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    if inp.is_dir():
        files = [str(p) for p in inp.rglob("*") if p.suffix.lower() in exts]
    else:
        files = [str(inp)]

    if not files:
        print("No audio files found.")
        return

    for f in files:
        try:
            out_png = process_file(f, out_dir, args.csv)
            print(f"Wrote {out_png}")
        except Exception as e:
            print(f"ERROR processing {f}: {e}")


if __name__ == "__main__":
    main()


