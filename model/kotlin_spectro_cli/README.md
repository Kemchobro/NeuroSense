# Kotlin Spectrogram Generator (CLI)

Generates 299x299 PNG spectrograms using the exact same Kotlin DSP code as the Android app (MainActivity.kt).
This ensures that the spectrograms generated for training match exactly what the app produces during inference.

## Requirements
- JDK 17+ (Java 23 is compatible and works fine)
- No need to install Gradle - the wrapper is included (`./gradlew`)

## Run

The Gradle wrapper is included, so you can run directly without installing Gradle:

```bash
cd model/kotlin_spectro_cli
# default inputs if args omitted: ../voice_dataset -> ../spectrograms_from_kotlin
./gradlew run --args "../voice_dataset ../spectrograms_from_kotlin"
```

Or if you have Gradle installed globally:
```bash
gradle run --args "../voice_dataset ../spectrograms_from_kotlin"
```

## Processing Details
- Reads from `voice_dataset/PD_AH` and `voice_dataset/HC_AH` directories
- Outputs to specified directory (default: `spectrograms_from_kotlin`) as `PD_####.png` / `HC_####.png`
- Only `.wav` (PCM 16-bit) files are processed
- Uses the exact same processing pipeline as MainActivity.kt:
  - Sample rate: 22050 Hz
  - n_fft: 2048
  - hop_length: 512
  - n_mels: 128
  - Mel scale: Slaney (librosa-compatible)
  - Normalization: Slaney (equal area per filter)
  - Power to dB: ref=max, amin=1e-10
  - Resize: Bilinear to 299x299
  - Output: Grayscale PNG images (min-max normalized)
