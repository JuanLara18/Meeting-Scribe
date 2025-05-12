#!/usr/bin/env python
# whisper_install.py - Modern installer for Whisper on Python 3.13
# ───────────────────────────────────────────────────────────────────
# This script handles installation of Whisper with the modern pyproject.toml structure

import os
import sys
import subprocess
import logging
import re
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("whisper_installer")

def print_header(text):
    """Print a section header with decorative elements."""
    logger.info("\n" + "="*70)
    logger.info(f"  {text}")
    logger.info("="*70)

def print_step(text):
    """Print a step header in the console."""
    logger.info(f"\n➤ {text}")

def print_success(text):
    """Print a success message."""
    logger.info(f"✓ {text}")

def print_error(text):
    """Print an error message to stderr."""
    logger.error(f"✗ {text}")

def run_command(cmd, check=True, capture_output=True, cwd=None):
    """
    Run a shell command and handle errors gracefully.
    
    Args:
        cmd: List of command and arguments
        check: Whether to raise an exception on failure
        capture_output: Whether to capture and return stdout/stderr
        cwd: Directory to run the command in
        
    Returns:
        CompletedProcess object from subprocess.run
    """
    try:
        print_step(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=cwd
        )
        if result.stdout and len(result.stdout) > 0:
            print(result.stdout[:500]) # Print first 500 chars of output
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print_error(f"Error output: {e.stderr[:500]}") # Print first 500 chars of error
        if not check:
            return e
        raise

def clone_whisper_repo(target_dir="whisper_src"):
    """
    Clone the Whisper repository to a local directory.
    
    Args:
        target_dir: Directory to clone into
        
    Returns:
        Path to the cloned repository
    """
    print_step(f"Cloning Whisper repository to {target_dir}...")
    
    repo_path = Path(target_dir).absolute()
    
    # Check if directory already exists
    if repo_path.exists():
        print_step(f"Directory {repo_path} already exists. Checking if it's a git repo...")
        if (repo_path / ".git").exists():
            # If it's a git repo, we'll pull the latest changes
            run_command(["git", "reset", "--hard", "HEAD"], cwd=repo_path)
            run_command(["git", "pull"], cwd=repo_path)
            print_success("Updated existing repository")
            return repo_path
        else:
            print_step("Not a git repository. Removing and recloning...")
            shutil.rmtree(repo_path)
    
    # Create directory if needed
    repo_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Clone the repository
    result = run_command(
        ["git", "clone", "https://github.com/openai/whisper.git", str(repo_path)],
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print_success("Repository cloned successfully")
        return repo_path
    else:
        print_error("Failed to clone repository")
        return None

def patch_pyproject_toml(repo_path):
    """
    Patch the pyproject.toml file to fix Python 3.13 compatibility.
    
    Args:
        repo_path: Path to the Whisper repository
        
    Returns:
        bool: Success or failure
    """
    print_step("Patching pyproject.toml for Python 3.13 compatibility...")
    
    pyproject_path = repo_path / "pyproject.toml"
    
    # Verify the file exists
    if not pyproject_path.exists():
        print_error(f"pyproject.toml not found at {pyproject_path}")
        return False
    
    try:
        # Read the file
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        # Make a backup
        with open(pyproject_path.with_suffix('.toml.bak'), 'w') as f:
            f.write(content)
        
        # Look for dynamic versioning
        if "dynamic = [" in content and "version" in content:
            # Remove "version" from dynamic list
            pattern = r'dynamic\s*=\s*\[(.*?)\]'
            match = re.search(pattern, content)
            if match:
                dynamic_items = match.group(1)
                # Remove "version" from the dynamic items
                new_dynamic_items = re.sub(r'["\']\s*version\s*["\']', '', dynamic_items)
                # Clean up any duplicate commas that might be left
                new_dynamic_items = re.sub(r',\s*,', ',', new_dynamic_items)
                # Remove leading/trailing commas
                new_dynamic_items = new_dynamic_items.strip(',').strip()
                # Update the dynamic list
                if new_dynamic_items:
                    content = re.sub(pattern, f'dynamic = [{new_dynamic_items}]', content)
                else:
                    # If dynamic list is empty, we might need to remove the whole line
                    content = re.sub(r'dynamic\s*=\s*\[\s*"version"\s*\]\s*\n', '', content)
        
        # Add static version if needed
        if 'version = ' not in content:
            # Find the [project] section
            if '[project]' in content:
                # Add version right after [project]
                content = content.replace('[project]', '[project]\nversion = "20240930"')
            else:
                print_error("Could not find [project] section in pyproject.toml")
                return False
        else:
            # Replace existing version line
            content = re.sub(
                r'version\s*=\s*\{.*?\}', 
                'version = "20240930"', 
                content
            )
        
        # Write the modified file
        with open(pyproject_path, 'w') as f:
            f.write(content)
        
        print_success("pyproject.toml patched successfully")
        
        # Also create an empty __version__.py file if it doesn't exist
        version_file = repo_path / "whisper" / "__version__.py"
        if not version_file.exists():
            version_file.parent.mkdir(exist_ok=True, parents=True)
            with open(version_file, 'w') as f:
                f.write('__version__ = "20240930"\n')
            print_success("Created __version__.py file")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to patch pyproject.toml: {str(e)}")
        return False

def install_whisper(repo_path):
    """
    Install Whisper from the local directory using different methods.
    
    Args:
        repo_path: Path to the patched Whisper repository
        
    Returns:
        bool: Success or failure
    """
    print_step("Installing Whisper package with no-build-isolation...")
    
    # Method 1: Install with no-build-isolation
    try:
        result = run_command(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
            check=False,
            cwd=repo_path
        )
        
        if result.returncode == 0:
            print_success("Installation succeeded with --no-build-isolation")
            return verify_installation()
    except Exception as e:
        print_error(f"First installation method failed: {str(e)}")
    
    # Method 2: Try direct pip install
    print_step("Trying alternative installation method...")
    try:
        result = run_command(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=False,
            cwd=repo_path
        )
        
        if result.returncode == 0:
            print_success("Installation succeeded with direct -e method")
            return verify_installation()
    except Exception as e:
        print_error(f"Second installation method failed: {str(e)}")
    
    # Method 3: Try pip install with --editable
    print_step("Trying final installation method...")
    try:
        result = run_command(
            [sys.executable, "-m", "pip", "install", "--editable", "."],
            check=False,
            cwd=repo_path
        )
        
        if result.returncode == 0:
            print_success("Installation succeeded with --editable method")
            return verify_installation()
        else:
            print_error("All installation methods failed")
            return False
    except Exception as e:
        print_error(f"All installation methods failed: {str(e)}")
        return False

def verify_installation():
    """
    Verify that Whisper was installed correctly.
    
    Returns:
        bool: Success or failure
    """
    print_step("Verifying Whisper installation...")
    try:
        result = run_command(
            [sys.executable, "-c", "import whisper; print(f'Whisper installed successfully!')"],
            check=False
        )
        
        if result.returncode == 0:
            print_success("Whisper verified successfully")
            return True
        else:
            print_error("Whisper verification failed")
            if result.stderr:
                print_error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Verification error: {str(e)}")
        return False

def create_version_file(repo_path):
    """
    Create a __version__.py file in the whisper package directory.
    
    Args:
        repo_path: Path to the Whisper repository
    """
    print_step("Creating version file...")
    
    whisper_dir = repo_path / "whisper"
    if not whisper_dir.exists() or not whisper_dir.is_dir():
        print_error(f"Whisper package directory not found at {whisper_dir}")
        return False
    
    version_file = whisper_dir / "__version__.py"
    try:
        with open(version_file, 'w') as f:
            f.write('__version__ = "20240930"\n')
        print_success(f"Created {version_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create version file: {str(e)}")
        return False

def fallback_install():
    """
    If all else fails, try direct pip install from the repository.
    
    Returns:
        bool: Success or failure
    """
    print_step("Trying fallback installation directly from GitHub...")
    
    try:
        result = run_command(
            [
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "git+https://github.com/openai/whisper.git@main"
            ],
            check=False
        )
        
        if result.returncode == 0:
            print_success("Fallback installation successful")
            return verify_installation()
        else:
            print_error("Fallback installation failed")
            return False
    except Exception as e:
        print_error(f"Fallback installation error: {str(e)}")
        return False

def main():
    """Main installation workflow"""
    print_header("Whisper Manual Installation (Python 3.13 Compatible)")
    
    try:
        # Show Python version
        python_version = sys.version.split()[0]
        print_step(f"Using Python {python_version}")
        
        # 1. Clone the repository
        repo_path = clone_whisper_repo()
        if not repo_path:
            return fallback_install()
        
        # 2. List the repository contents to help debug
        print_step(f"Repository contents: {os.listdir(repo_path)}")
        
        # 3. Patch the pyproject.toml file
        if not patch_pyproject_toml(repo_path):
            print_error("Failed to patch configuration file")
            return fallback_install()
        
        # 4. Create version file as a fallback
        create_version_file(repo_path)
        
        # 5. Install the package
        if not install_whisper(repo_path):
            return fallback_install()
            
        print_header("Whisper Installation Complete!")
        print_step("You can now run MeetingScribe with: python main.py path/to/video.mp4")
        return True
        
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return fallback_install()

if __name__ == "__main__":
    success = main()
    
    if not success:
        print_header("Whisper Installation Failed")
        print_step("Please try installing a compatible alternative:")
        print("  pip install faster-whisper")
        print("\nThen modify the processing/transcribe.py file to use faster-whisper")
        print("\nOr try installing whisper manually with:")
        print("  pip install git+https://github.com/openai/whisper.git@main")
        sys.exit(1)
    
    sys.exit(0)