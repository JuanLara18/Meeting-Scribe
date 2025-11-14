# processing/transcribe.py
# ───────────────────────────────────────────────────────────────────
# Wraps the Whisper ASR model to convert WAV audio into timestamped
# text segments (raw JSON + Python dicts).

import os
import json
import logging
import time
from typing import Dict, Any, Optional

import whisper

class WhisperTranscriber:
    """
    Transcribes speech from audio into text segments using OpenAI Whisper.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = None,
        verbose: bool = False,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0
    ):
        """
        Args:
          model_size – one of ["tiny","base","small","medium","large-v3"]
          device     – "cpu" or "cuda" (GPU)
          language   – ISO-639-1 code to force transcription language, or None
          verbose    – True to log detailed progress
          temperature – sampling temperature (0 = greedy, higher = more random)
          beam_size  – beam search width (higher = better quality, slower)
          best_of    – number of candidates when sampling (higher = better)
          patience   – beam search patience factor
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.verbose = verbose
        self.temperature = temperature
        self.beam_size = beam_size
        self.best_of = best_of
        self.patience = patience
        self.logger = logging.getLogger(__name__)
        self.model = None  # Lazy loading - model will be loaded on first use
        
        # Configure logging based on verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def _load_model(self) -> None:
        """
        Load the Whisper model if not already loaded.
        """
        if self.model is None:
            valid_sizes = ["tiny", "base", "small", "medium", "large-v3"]
            if self.model_size not in valid_sizes:
                self.logger.warning(f"Invalid model size: {self.model_size}. Using 'base' instead.")
                self.model_size = "base"
            
            self.logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            start_time = time.time()
            
            try:
                self.model = whisper.load_model(self.model_size, device=self.device)
                load_time = time.time() - start_time
                self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {str(e)}")
                raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def transcribe(
        self,
        wav_path: str,
        output_json: bool = False,
        initial_prompt: str = None,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Run Whisper on the given WAV file.

        Args:
          wav_path         – path to 16 kHz mono WAV
          output_json      – if True, save raw whisper JSON to disk
          initial_prompt   – optional text to guide model style/terminology
          word_timestamps  – if True, extract word-level timestamps

        Returns:
          A dict matching Whisper's JSON output, e.g.:
            {
              "text":    "full transcript",
              "segments":[
                  {"id": 0, "start":0.0, "end":2.1, "text":"Hello ...", "words": [...]},
                  …
              ]
            }

        Raises:
          FileNotFoundError if wav_path missing
          RuntimeError on model load or inference error
        """
        # Check if input file exists
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")

        # Load model if not already loaded
        self._load_model()

        # Prepare transcription options with improved quality settings
        transcribe_options = {
            "temperature": self.temperature,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "condition_on_previous_text": True,  # Better context consistency
            "word_timestamps": word_timestamps,   # Extract word-level timestamps
        }

        # Add language constraint if specified
        if self.language:
            transcribe_options["language"] = self.language

        # Add initial prompt if provided (helps with terminology and style)
        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt

        # Log start of transcription
        self.logger.info(f"Starting transcription of {wav_path}")
        if self.verbose:
            self.logger.debug(f"Transcription options: {transcribe_options}")

        start_time = time.time()

        try:
            # Run Whisper transcription
            result = self.model.transcribe(wav_path, **transcribe_options)

            # Log completion
            transcribe_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {transcribe_time:.2f} seconds")

            if self.verbose:
                num_segments = len(result.get("segments", []))
                total_duration = result["segments"][-1]["end"] if num_segments > 0 else 0
                self.logger.debug(f"Generated {num_segments} segments, total duration: {total_duration:.2f}s")

                # Log word count if word timestamps are available
                if word_timestamps and num_segments > 0:
                    total_words = sum(len(seg.get("words", [])) for seg in result["segments"])
                    if total_words > 0:
                        self.logger.debug(f"Extracted {total_words} word-level timestamps")

            # Save JSON output if requested
            if output_json:
                json_path = os.path.splitext(wav_path)[0] + "_transcript.json"
                self._save_json(result, json_path)

            return result

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Whisper transcription failed: {str(e)}")
    
    def _save_json(self, result: Dict[str, Any], json_path: str) -> None:
        """
        Save transcription result to JSON file.
        
        Args:
            result: Whisper transcription result
            json_path: Path where JSON will be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
            
            # Write JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved transcript JSON to {json_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON output: {str(e)}")