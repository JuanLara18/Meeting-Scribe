# processing/audio.py
# ───────────────────────────────────────────────────────────────────
# Handles conversion of video files into clean, resampled WAV audio
# using ffmpeg, ready for ASR and diarization.

import os
import subprocess
import logging
from typing import List, Optional

class AudioExtractor:
    """
    Handles video-to-audio conversion using ffmpeg.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Args:
          ffmpeg_path – executable name or full path to ffmpeg binary
        """
        self.ffmpeg_path = ffmpeg_path
        self.logger = logging.getLogger(__name__)

    def _check_ffmpeg_available(self) -> bool:
        """
        Verify ffmpeg is callable using the configured path.
        
        Returns:
            True if ffmpeg is available, False otherwise.
        """
        try:
            # Run ffmpeg with -version to check if it's available and working
            result = subprocess.run(
                [self.ffmpeg_path, "-version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            self.logger.error(f"ffmpeg not found at path: {self.ffmpeg_path}")
            return False

    def extract(
        self,
        video_path: str,
        target_wav: str,
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> str:
        """
        Invoke ffmpeg to produce a WAV file suitable for ASR.

        Args:
          video_path – input video file
          target_wav – output path for .wav
          sample_rate– e.g. 16000
          mono       – True to downmix to single channel

        Returns:
          path to the generated WAV file (same as target_wav)

        Raises:
          FileNotFoundError if video_path is missing
          RuntimeError on ffmpeg failure
        """
        # Check if input file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg_available():
            raise RuntimeError(f"ffmpeg not found or not executable at: {self.ffmpeg_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(target_wav)), exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,     # Input file
            "-vn",                # Disable video
            "-ar", str(sample_rate),  # Audio sample rate
        ]
        
        # Add mono/stereo option
        if mono:
            cmd.extend(["-ac", "1"])  # Mono audio (1 channel)
        
        # Add remaining parameters
        cmd.extend([
            "-c:a", "pcm_s16le",  # 16-bit PCM codec
            "-y",                 # Overwrite output file
            target_wav            # Output file
        ])
        
        self.logger.info(f"Converting video to audio: {video_path} -> {target_wav}")
        self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        
        try:
            # Run ffmpeg command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check if ffmpeg completed successfully
            if process.returncode != 0:
                error_msg = process.stderr or "No error details available"
                self.logger.error(f"ffmpeg failed with code {process.returncode}: {error_msg}")
                raise RuntimeError(f"ffmpeg conversion failed: {error_msg}")
            
            # Check if output file was created
            if not os.path.isfile(target_wav):
                raise RuntimeError(f"ffmpeg did not create output file: {target_wav}")
            
            self.logger.info(f"Successfully extracted audio to {target_wav}")
            return target_wav
            
        except Exception as e:
            self.logger.error(f"Error during audio extraction: {str(e)}")
            raise RuntimeError(f"Error during audio extraction: {str(e)}")