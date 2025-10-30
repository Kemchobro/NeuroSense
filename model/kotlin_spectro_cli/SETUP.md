# Setup Instructions

## Quick Setup

You have Java 23 installed, which is compatible with Java 17. The build is now configured to use your system Java.

### Option 1: Install Gradle via Homebrew (Recommended)

```bash
brew install gradle
```

Then run:
```bash
cd model/kotlin_spectro_cli
gradle run --args "../voice_dataset ../spectrograms_from_kotlin"
```

### Option 2: Use Gradle Wrapper

If you prefer to use the Gradle wrapper, you'll need to initialize it first. The wrapper files have been created, but you need Gradle installed to bootstrap it.

After installing Gradle (via Option 1), run:
```bash
cd model/kotlin_spectro_cli
gradle wrapper
./gradlew run --args "../voice_dataset ../spectrograms_from_kotlin"
```

### Option 3: Download Gradle Manually

1. Download Gradle 8.12 from https://gradle.org/releases/
2. Extract it and add to your PATH
3. Run: `gradle run --args "../voice_dataset ../spectrograms_from_kotlin"`

## What Changed

- Updated `build.gradle.kts` to remove the strict Java 17 toolchain requirement
- The build now uses your system Java (Java 23) which is backward compatible
- Code targets Java 17 bytecode for compatibility

