# whisper_install.py - Direct installer for Whisper on Python 3.13
# ───────────────────────────────────────────────────────────────────
# This script provides a direct installation method for Whisper when
# normal pip installation fails (especially on Python 3.13)

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def print_step(text):
    print(f"\n➤ {text}")

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}", file=sys.stderr)

def main():
    print_step("Installing Whisper directly (Python 3.13 compatible method)")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Working in temporary directory: {temp_path}")
        
        # Clone the repository
        print_step("Cloning Whisper repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/openai/whisper.git", temp_path],
                check=True,
                capture_output=True
            )
            print_success("Repository cloned")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to clone repository: {e}")
            print("Error output:", e.stderr.decode())
            return False
        
        # Modify setup.py to fix the version issue
        print_step("Patching setup.py for Python 3.13 compatibility...")
        setup_path = temp_path / "setup.py"
        
        try:
            with open(setup_path, 'r') as f:
                setup_content = f.read()
            
            # Replace the problematic version detection
            modified_content = setup_content.replace(
                'version=read_version()',
                'version="20240930"'  # Hardcode a specific version
            )
            
            with open(setup_path, 'w') as f:
                f.write(modified_content)
            
            print_success("setup.py patched")
        except Exception as e:
            print_error(f"Failed to patch setup.py: {e}")
            return False
        
        # Install the package
        print_step("Installing Whisper package...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(temp_path)],
                check=True
            )
            print_success("Whisper installed successfully")
            
            # Verify installation
            try:
                subprocess.run(
                    [sys.executable, "-c", "import whisper; print(f'Whisper version: {whisper.__version__}')"],
                    check=True
                )
                print_success("Whisper import verified")
                return True
            except subprocess.CalledProcessError:
                print_error("Whisper installed but import verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install Whisper: {e}")
            return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Whisper installation complete!")
        print("You can now run MeetingScribe with: python main.py path/to/video.mp4")
    else:
        print("\n❌ Whisper installation failed.")
        print("Try the alternative installation methods in INSTALL.md")
        sys.exit(1)