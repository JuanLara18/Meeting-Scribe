# processing/diarize.py
# ───────────────────────────────────────────────────────────────────
# Uses a pyannote.audio pipeline to segment audio by speaker,
# producing "who spoke when" time intervals.

import os
import logging
import time
import torch
from typing import List, Dict, Any, Optional

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from huggingface_hub import login

class SpeakerDiarizer:
    """
    Segments audio into speaker turns using pyannote.audio pipelines.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization",
        use_auth_token: str = None,
        device: str = "cpu"
    ):
        """
        Args:
          model_name    – Hugging Face model ID for diarization
          use_auth_token– optional HF token if model is gated
          device        – "cpu" or "cuda"
        """
        self.model_name = model_name
        self.use_auth_token = use_auth_token
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.pipeline = None  # Lazy loading - pipeline will be loaded on first use
    
    def _load_pipeline(self) -> None:
        """
        Load the diarization pipeline if not already loaded.
        
        Raises:
            RuntimeError if pipeline cannot be loaded
        """
        if self.pipeline is None:
            self.logger.info(f"Loading diarization pipeline {self.model_name} on {self.device}...")
            start_time = time.time()
            
            try:
                # If a token is provided, log in to Hugging Face
                if self.use_auth_token:
                    login(token=self.use_auth_token)
                    self.logger.debug("Authenticated with Hugging Face Hub")
                
                # Load the pipeline
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name, 
                    use_auth_token=self.use_auth_token
                )
                
                # Move to specified device
                self.pipeline.to(torch_device=self.device)
                
                load_time = time.time() - start_time
                self.logger.info(f"Pipeline loaded in {load_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to load diarization pipeline: {str(e)}")
                raise RuntimeError(f"Failed to load diarization pipeline: {str(e)}")

    def diarize(
        self,
        wav_path: str,
        min_speakers: int = None,
        max_speakers: int = None
    ) -> List[Dict[str, Any]]:
        """
        Run speaker diarization.

        Args:
          wav_path     – path to 16 kHz mono WAV
          min_speakers – optional lower bound on # of speakers
          max_speakers – optional upper bound on # of speakers

        Returns:
          A list of speaker segments:
            [
              {
                "speaker": "SPEAKER_00",
                "start":   0.00,
                "end":     1.75
              },
              …
            ]

        Raises:
          FileNotFoundError if wav_path missing
          RuntimeError on pipeline failure
        """
        # Check if input file exists
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")
        
        # Load pipeline if not already loaded
        self._load_pipeline()
        
        # Prepare diarization options
        diarize_options = {}
        
        # Add speaker constraints if specified
        if min_speakers is not None:
            diarize_options["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_options["max_speakers"] = max_speakers
        
        # Log start of diarization
        self.logger.info(f"Starting diarization of {wav_path}")
        self.logger.debug(f"Diarization options: {diarize_options}")
        
        start_time = time.time()
        
        try:
            # Create a progress hook if logging is enabled
            with ProgressHook() as hook:
                # Run diarization pipeline
                diarization = self.pipeline(wav_path, hooks=[hook], **diarize_options)
            
            # Convert pyannote format to our segment format
            segments = self._convert_to_segments(diarization)
            
            # Log completion
            diarize_time = time.time() - start_time
            self.logger.info(f"Diarization completed in {diarize_time:.2f} seconds")
            self.logger.info(f"Found {len(set(s['speaker'] for s in segments))} speakers across {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise RuntimeError(f"Pyannote diarization failed: {str(e)}")
    
    def _convert_to_segments(self, diarization) -> List[Dict[str, Any]]:
        """
        Convert pyannote.audio diarization output to a list of speaker segments.
        
        Args:
            diarization: pyannote.audio diarization result
            
        Returns:
            List of speaker segments in our format
        """
        segments = []
        
        # Iterate through the speaker turns
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            }
            segments.append(segment)
        
        # Sort segments by start time
        segments.sort(key=lambda s: s["start"])
        
        return segments