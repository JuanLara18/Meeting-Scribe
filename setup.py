#!/usr/bin/env python
# setup.py - Environment setup helper for MeetingScribe
# ───────────────────────────────────────────────────────────────────
# Run this script to set up your environment for MeetingScribe

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header(text):
    """Print a section header."""
    print(f"\n{'=' * 50}")
    print(f"  {text}")
    print(f"{'=' * 50}\n")

def print_step(text):
    """Print a step message."""
    print(f"➤ {text}")

def print_success(text):
    """Print a success message."""
    print(f"✓ {text}")

def print_error(text):
    """Print an error message."""
    print(f"✗ {text}", file=sys.stderr)

def check_python_version():
    """Check if Python version is at least 3.10."""
    print_step("Checking Python version...")
    
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 10):
        print_error(f"Python 3.10+ is required, but you have {major}.{minor}")
        return False
    
    print_success(f"Python {major}.{minor} detected")
    return True

def check_ffmpeg():
    """Check if ffmpeg is installed and in PATH."""
    print_step("Checking ffmpeg installation...")
    
    if shutil.which("ffmpeg") is None:
        print_error("ffmpeg not found in PATH")
        
        # Provide installation instructions based on OS
        system = platform.system().lower()
        if system == "linux":
            print("  Install with: sudo apt install ffmpeg (Debian/Ubuntu)")
            print("  or: sudo dnf install ffmpeg (Fedora)")
        elif system == "darwin":
            print("  Install with: brew install ffmpeg (using Homebrew)")
        elif system == "windows":
            print("  Download from: https://ffmpeg.org/download.html")
            print("  Or install with: choco install ffmpeg (using Chocolatey)")
        
        return False
    
    # Check version
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True
        )
        version_line = result.stdout.split('\n')[0]
        print_success(f"ffmpeg detected: {version_line}")
        return True
    except Exception as e:
        print_error(f"Error getting ffmpeg version: {e}")
        return False

def create_virtual_env():
    """Create a virtual environment if it doesn't exist."""
    print_step("Setting up virtual environment...")
    
    venv_dir = Path(".venv")
    if venv_dir.exists():
        print_success("Virtual environment already exists")
        return True
    
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", ".venv"],
            check=True
        )
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install Python dependencies using pip."""
    print_step("Installing Python dependencies...")
    
    # Determine pip path based on platform
    system = platform.system().lower()
    if system == "windows":
        pip_path = Path(".venv") / "Scripts" / "pip"
    else:
        pip_path = Path(".venv") / "bin" / "pip"
    
    try:
        # Upgrade pip first
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            check=True
        )
        
        # Install core dependencies first (except whisper)
        print("Installing core packages...")
        
        # Basic dependencies that should be stable
        core_packages = [
            "numpy>=1.24.0",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "ffmpeg-python>=0.2.0",
            "tqdm>=4.65.0",
            "click>=8.1.3",
            "rich>=13.3.5",
            "python-dotenv>=1.0.0"
        ]
        
        for package in core_packages:
            try:
                subprocess.run(
                    [str(pip_path), "install", package],
                    check=True
                )
                print(f"  Installed {package.split('>=')[0]}")
            except subprocess.CalledProcessError:
                print(f"  Failed to install {package}")
        
        # Install whisper directly from GitHub
        print("\nInstalling Whisper (this may take a few minutes)...")
        whisper_repo = "git+https://github.com/openai/whisper.git@248b6cb124225dd263bb9bd32d060b6517e067f8"
        try:
            subprocess.run(
                [str(pip_path), "install", whisper_repo],
                check=True
            )
            print_success("Whisper installed")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install Whisper: {e}")
            print("You may need to install it manually after setup.")
        
        # Install remaining packages from requirements
        print("\nInstalling remaining dependencies...")
        try:
            subprocess.run(
                [str(pip_path), "install", "-r", "requirements.txt"],
                check=False  # Don't fail if some packages are already installed or fail
            )
        except subprocess.CalledProcessError:
            print("Not all dependencies could be installed.")
            print("Some advanced features may not work correctly.")
        
        print_success("Core dependencies installed")
        return True
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def setup_huggingface_auth():
    """Set up Hugging Face authentication for pyannote.audio."""
    print_step("Setting up Hugging Face authentication...")
    
    # Check if token is already in environment
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\nNOTE: pyannote.audio speaker diarization models require authentication with Hugging Face.")
        print("You need to:")
        print("1. Create an account at https://huggingface.co/")
        print("2. Accept the license agreement at: https://huggingface.co/pyannote/speaker-diarization")
        print("3. Get your API token from: https://huggingface.co/settings/tokens\n")
        
        token = input("Enter your Hugging Face token (or press Enter to skip for now): ").strip()
        
        if token:
            # Create .env file for storing the token
            with open(".env", "w") as f:
                f.write(f"HF_TOKEN={token}\n")
            print_success("Token saved to .env file")
        else:
            print("Skipping Hugging Face authentication for now.")
            print("You can add your token to a .env file later with: HF_TOKEN=your_token_here")
    else:
        print_success("Hugging Face token found in environment")
    
    return True

def main():
    """Main setup function."""
    print_header("MeetingScribe Setup")
    
    # Run all checks and setup steps
    steps = [
        check_python_version,
        check_ffmpeg,
        create_virtual_env,
        install_dependencies,
        setup_huggingface_auth
    ]
    
    success = True
    for step in steps:
        if not step():
            success = False
            break
    
    if success:
        print_header("Setup Complete!")
        
        # Determine activation command based on platform
        system = platform.system().lower()
        if system == "windows":
            activate_cmd = r".\.venv\Scripts\activate"
        else:
            activate_cmd = "source .venv/bin/activate"
        
        print(f"To activate the environment, run: {activate_cmd}")
        print("Then run MeetingScribe with: python main.py path/to/video.mp4")
    else:
        print_header("Setup Failed")
        print("Please fix the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()