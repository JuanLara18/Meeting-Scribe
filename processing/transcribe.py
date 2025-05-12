# processing/transcribe.py
# ───────────────────────────────────────────────────────────────────
# Wraps the Whisper ASR model to convert WAV audio into timestamped
# text segments (raw JSON + Python dicts).

from typing import Dict, Any

class WhisperTranscriber:
    """
    Transcribes speech from audio into text segments using OpenAI Whisper.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = None,
        verbose: bool = False
    ):
        """
        Args:
          model_size – one of ["tiny","base","small","medium","large-v3"]
          device     – "cpu" or "cuda" (GPU)
          language   – ISO-639-1 code to force transcription language, or None
          verbose    – True to log detailed progress
        """
        ...

    def transcribe(
        self,
        wav_path: str,
        output_json: bool = False
    ) -> Dict[str, Any]:
        """
        Run Whisper on the given WAV file.

        Args:
          wav_path     – path to 16 kHz mono WAV
          output_json  – if True, save raw whisper JSON to disk

        Returns:
          A dict matching Whisper’s JSON output, e.g.:
            {
              "text":    "full transcript",
              "segments":[
                  {"id": 0, "start":0.0, "end":2.1, "text":"Hello ..."},
                  …
              ]
            }

        Raises:
          FileNotFoundError if wav_path missing
          RuntimeError on model load or inference error
        """
        ...
