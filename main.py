# main.py
# ───────────────────────────────────────────────────────────────────
# Orchestrates the full pipeline: extract audio, transcribe speech,
# diarize speakers, merge segments, and export results to JSON/Markdown.


class MeetingScribe:
    """
    Orchestrates the end-to-end pipeline:
      1) Extract audio from video
      2) Transcribe audio
      3) Diarize speakers
      4) Merge segments
      5) Export Markdown & JSON
    """

    def __init__(
        self,
        video_path: str,
        output_folder: str = "results/",
        language: str = None,
        whisper_model: str = "base",
    ):
        """
        Args:
          video_path    – path to input video file (.mp4/.mkv/.mov)
          output_folder – where transcript.json and .md will be created
          language      – ISO-639-1 code to force ASR language (None=auto)
          whisper_model – one of ["tiny","base","small","medium","large-v3"]
        """
        ...

    def run(self) -> None:
        """
        Execute full pipeline in sequence.
        Raises exception on any step failure.
        """

    # Internal helpers—callable by run()
    def _extract_audio(self) -> str:
        """
        Returns:
          path to extracted .wav file (16 kHz, mono)
        """

    def _transcribe(self, wav_path: str) -> dict:
        """
        Args:
          wav_path – path to .wav file from _extract_audio()
        Returns:
          Whisper transcript dict with segments & timestamps
        """

    def _diarize(self, wav_path: str) -> list:
        """
        Args:
          wav_path – same .wav file
        Returns:
          list of diarization segments, each with:
            - speaker_id: str
            - start_time: float (secs)
            - end_time: float   (secs)
        """

    def _merge(self, transcript: dict, diarization: list) -> list:
        """
        Align transcript segments with speaker turns.
        Args:
          transcript   – output of _transcribe()
          diarization  – output of _diarize()
        Returns:
          list of merged segments:
            [{
              "speaker": "...",
              "start": 12.34,
              "end":   15.67,
              "text":  "…"
            }, …]
        """

    def _export(self, merged: list) -> None:
        """
        Writes:
          - results/transcript.json (raw merged list)
          - results/transcript.md   (Markdown with headers per minute)
        """
