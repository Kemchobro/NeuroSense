# ===============================================================
# spectrogram_service.py
# Chaquopy-compatible Mel spectrogram generator for Flutter Android
# ===============================================================

import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless Android environments
import matplotlib.pyplot as plt
import time

# Configuration constants
SAMPLE_RATE = 22050
N_MELS = 128
OUTPUT_DIR = "spectrogram_output"

# Ensure directory exists inside app storage
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_spectrogram(audio_path):
    """
    Generate and save a mel spectrogram image from a .wav file.
    Returns the absolute path of the generated image.
    """
    try:
        start_time = time.time()
        if not os.path.exists(audio_path):
            return f"ERROR: File not found -> {audio_path}"

        # Load the audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return "ERROR: Empty or invalid audio file."

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot and save image
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(mel_db, sr=sr, cmap="magma")
        plt.axis("off")
        plt.tight_layout(pad=0)

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_spectrogram.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        elapsed = round(time.time() - start_time, 2)
        return f"SUCCESS: {os.path.abspath(output_path)} (in {elapsed}s)"

    except Exception as e:
        return f"ERROR: {str(e)}"
