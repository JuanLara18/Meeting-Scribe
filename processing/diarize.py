# processing/diarize.py
# ───────────────────────────────────────────────────────────────────
# Uses a pyannote.audio pipeline to segment audio by speaker,
# producing “who spoke when” time intervals.


from typing import List, Dict, Any

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
        ...

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
        ...
