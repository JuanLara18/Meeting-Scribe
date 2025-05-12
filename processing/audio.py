# processing/audio.py
# ───────────────────────────────────────────────────────────────────
# Handles conversion of video files into clean, resampled WAV audio
# using ffmpeg, ready for ASR and diarization.


class AudioExtractor:
    """
    Handles video-to-audio conversion using ffmpeg.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Args:
          ffmpeg_path – executable name or full path to ffmpeg binary
        """

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
