# processing/merge.py
# ───────────────────────────────────────────────────────────────────
# Aligns Whisper’s transcript segments with diarization turns,
# producing unified speaker-labeled text entries.

from typing import List, Dict, Any

class SegmentMerger:
    """
    Merges ASR segments and diarization segments into unified speaker-labelled entries.
    """

    def __init__(
        self,
        max_gap: float = 0.5,
        min_overlap: float = 0.1
    ):
        """
        Args:
          max_gap     – maximum seconds allowed between ASR and speaker turn to still align
          min_overlap – minimum overlap in seconds required to consider segments matching
        """
        ...

    def merge(
        self,
        transcript_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aligns and merges two lists of segments.

        Args:
          transcript_segments – Whisper output `.segments`, each:
            {"id": int, "start": float, "end": float, "text": str}
          diarization_segments – pyannote output, each:
            {"speaker": str, "start": float, "end": float}

        Returns:
          A list of merged entries:
          [
            {
              "speaker":  "SPEAKER_00",
              "start":    12.34,
              "end":      15.67,
              "text":     "Transcribed text..."
            },
            …
          ]

        Raises:
          ValueError if inputs are empty or cannot be aligned
        """
        ...
