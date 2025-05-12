# Installation Guide for MeetingScribe

This guide provides detailed instructions for setting up MeetingScribe on different operating systems.

## Prerequisites

### Python 3.10 - 3.12 (Recommended)

MeetingScribe works best with Python 3.10, 3.11, or 3.12. Python 3.13 is supported but may require manual dependency installation. Verify your Python version with:

```bash
python --version
```

If needed, download Python from [python.org](https://www.python.org/downloads/).

### FFmpeg

FFmpeg is required for audio/video processing. Installation varies by platform:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

Or download directly from [ffmpeg.org](https://ffmpeg.org/download.html).

Verify installation with:
```bash
ffmpeg -version
```

## Installation Options

### Option 1: Automated Setup (Recommended)

For a guided setup experience, run:

```bash
# Clone the repository
git clone https://github.com/your-user/meetingscribe.git
cd meetingscribe

# Run the setup script
python setup.py
```

The script will:
1. Check Python version and FFmpeg installation
2. Create a virtual environment
3. Install all dependencies
4. Guide you through Hugging Face authentication

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-user/meetingscribe.git
cd meetingscribe

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Speaker Diarization Setup

The speaker diarization model requires authentication with Hugging Face:

1. Create an account at [huggingface.co](https://huggingface.co/)
2. Accept the license agreement at [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
3. Get your API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Create a `.env` file in the project root with:
   ```
   HF_TOKEN=your_token_here
   ```

## GPU Acceleration (Optional)

For faster processing on compatible hardware:

1. Install CUDA and cuDNN (see [PyTorch documentation](https://pytorch.org/get-started/locally/))
2. Uncomment the CUDA-specific lines in `requirements.txt`
3. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Troubleshooting

### Whisper Installation Issues

If you encounter issues installing the Whisper package, try these steps:

```bash
# Activate your virtual environment first, then:
pip install git+https://github.com/openai/whisper.git@248b6cb124225dd263bb9bd32d060b6517e067f8
```

## Python 3.13 Specific Instructions

**Using Python 3.13 requires special handling for the Whisper package.**

After running the basic setup, use our dedicated installer script:

```bash
# After initial setup.py completes (even with Whisper errors):
python whisper_install.py
```

This script will:
1. Clone the Whisper repository
2. Patch it for Python 3.13 compatibility 
3. Install it directly

If this fails, try the manual approach:

```bash
# Clone whisper
git clone https://github.com/openai/whisper.git
cd whisper

# Edit setup.py - replace "version=read_version()" with "version='20240930'"
# (Use any text editor to make this change)

# Install from local directory
pip install -e .
```

### Missing diarization models

If you encounter errors about missing diarization models:

1. Ensure you've set up your Hugging Face token as described above
2. Try manually downloading the model:
```bash
# Activate your virtual environment, then:
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token='YOUR_TOKEN_HERE')"
```

### Other issues

For other dependency issues, try installing the minimal set needed to run:

```bash
pip install torch torchaudio ffmpeg-python rich
pip install git+https://github.com/openai/whisper.git@248b6cb124225dd263bb9bd32d060b6517e067f8
pip install pyannote.audio
```

## Verification

Test your installation with:

```bash
python main.py --help
```

You should see the command-line help information for MeetingScribe.