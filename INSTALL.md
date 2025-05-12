# Installation Guide for MeetingScribe

This guide provides detailed instructions for setting up MeetingScribe on different operating systems, with special focus on Python 3.13 compatibility.

## Prerequisites

### Python 3.10 - 3.13

MeetingScribe works with Python 3.10, 3.11, 3.12, and 3.13. Verify your Python version with:

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
3. Install most dependencies
4. Guide you through Hugging Face authentication

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/JuanLara18/meetingscribe.git
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

## Python 3.13 Installation (Special Instructions)

**For Python 3.13 users, the Whisper installation requires a special approach**

After running the basic setup (which may show Whisper installation errors), use our dedicated installer script:

```bash
# After initial setup.py completes (even with Whisper errors):
python whisper_install.py
```

This script will:
1. Clone the Whisper repository
2. Patch its configuration files for Python 3.13 compatibility
3. Install it directly using the appropriate method

The script tries multiple installation methods and includes fallbacks if needed.

### Whisper Installation Troubleshooting

If the automated script fails, you can try manual installation:

1. Clone the repository:
```bash
git clone https://github.com/openai/whisper.git
cd whisper
```

2. Edit `pyproject.toml`:
   - Find any dynamic version settings like `dynamic = ["version"]` or `version = {attr = ...}`
   - Replace with `version = "20240930"`

3. Create a version file:
```bash
# Create whisper/__version__.py with this content:
echo '__version__ = "20240930"' > whisper/__version__.py
```

4. Install with build isolation disabled:
```bash
pip install --no-build-isolation -e .
```

5. Verify installation:
```bash
python -c "import whisper; print('Whisper installed!')"
```

### Alternative: Using faster-whisper

If you continue to face issues with Whisper installation, you can use faster-whisper as an alternative:

```bash
pip install faster-whisper
```

Then modify `processing/transcribe.py` to use faster-whisper instead of OpenAI's Whisper. This change will require some code adaptations but can work as a replacement.

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

## Common Problems and Solutions

### Missing diarization models

If you encounter errors about missing diarization models:

1. Ensure you've set up your Hugging Face token as described above
2. Try manually downloading the model:
```bash
# Activate your virtual environment, then:
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token='YOUR_TOKEN_HERE')"
```

### Whisper module import errors

If you see import errors after installing Whisper:

1. Make sure your virtual environment is activated
2. Try reinstalling with the direct GitHub URL:
```bash
pip install git+https://github.com/openai/whisper.git@main
```

### FFmpeg-related errors

If you see errors related to FFmpeg:

1. Verify FFmpeg is installed with `ffmpeg -version`
2. Ensure it's in your system PATH
3. On Windows, you might need to restart your terminal after installing FFmpeg

## Verification

Test your installation with:

```bash
python main.py --help
```

You should see the command-line help information for MeetingScribe.

## Running MeetingScribe

After successful installation:

```bash
python main.py path/to/your/video.mp4
```

The transcription and diarization results will be saved in the `results/` folder.