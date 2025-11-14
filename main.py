# main.py
# ───────────────────────────────────────────────────────────────────
# Orchestrates the full pipeline: extract audio, transcribe speech,
# diarize speakers, merge segments, and export results to JSON/Markdown.

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

import torch

from processing.audio import AudioExtractor
from processing.transcribe import WhisperTranscriber
from processing.diarize import SpeakerDiarizer
from processing.merge import SegmentMerger
from utils.markdown import MarkdownExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

def get_device(device: str = "auto") -> str:
    """
    Determine the compute device to use.

    Args:
        device: One of "auto", "cpu", "cuda", or "mps" (for Apple Silicon)

    Returns:
        Device string ("cpu", "cuda", or "mps")
    """
    logger = logging.getLogger(__name__)

    if device == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Silicon GPU (MPS) detected")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
    else:
        # Validate user-specified device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to CPU")
            device = "cpu"

    logger.info(f"Using device: {device}")
    return device

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
        whisper_model: str = "small",
        device: str = "auto",
    ):
        """
        Args:
          video_path    – path to input video file (.mp4/.mkv/.mov)
          output_folder – where transcript.json and .md will be created
          language      – ISO-639-1 code to force ASR language (None=auto)
          whisper_model – one of ["tiny","base","small","medium","large-v3"] (default: small)
          device        – compute device: "auto", "cpu", "cuda", or "mps"
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.language = language
        self.whisper_model = whisper_model
        self.device = get_device(device)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output folder ends with a slash for path consistency
        if not self.output_folder.endswith("/") and not self.output_folder.endswith("\\"):
            self.output_folder += "/"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Set paths for outputs
        self.audio_path = os.path.join(self.output_folder, "audio.wav")
        self.transcript_json = os.path.join(self.output_folder, "transcript.json")
        self.transcript_md = os.path.join(self.output_folder, "transcript.md")
        
        # Validate video path
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Validate model name
        valid_models = ["tiny", "base", "small", "medium", "large-v3"]
        if whisper_model not in valid_models:
            raise ValueError(f"Invalid model: {whisper_model}. Must be one of {valid_models}")

    def run(self) -> None:
        """
        Execute full pipeline in sequence.
        Raises exception on any step failure.
        """
        self.logger.info(f"Starting MeetingScribe pipeline for {self.video_path}")
        self.logger.info(f"Output folder: {self.output_folder}")
        
        try:
            # Step 1: Extract audio from video
            wav_path = self._extract_audio()
            
            # Step 2: Transcribe audio to text
            transcript = self._transcribe(wav_path)
            
            # Step 3: Diarize speakers in audio
            diarization = self._diarize(wav_path)
            
            # Step 4: Merge transcript and speaker info
            merged = self._merge(transcript, diarization)
            
            # Step 5: Export results to Markdown and JSON
            self._export(merged)
            
            self.logger.info("✓ Pipeline completed successfully")
            self.logger.info(f"📄 Results saved to {self.transcript_md} and {self.transcript_json}")
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _extract_audio(self) -> str:
        """
        Returns:
          path to extracted .wav file (16 kHz, mono)
        """
        self.logger.info("Step 1: Extracting audio from video")
        
        try:
            extractor = AudioExtractor()
            wav_path = extractor.extract(
                video_path=self.video_path,
                target_wav=self.audio_path,
                sample_rate=16000,  # 16kHz for optimal ASR
                mono=True           # Single channel for diarization
            )
            
            self.logger.info(f"Audio extracted to {wav_path}")
            return wav_path
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {str(e)}")
            raise RuntimeError(f"Audio extraction failed: {str(e)}")

    def _transcribe(self, wav_path: str) -> Dict[str, Any]:
        """
        Args:
          wav_path – path to .wav file from _extract_audio()
        Returns:
          Whisper transcript dict with segments & timestamps
        """
        self.logger.info("Step 2: Transcribing audio with Whisper")
        self.logger.info(f"Using model: {self.whisper_model}, language: {self.language or 'auto-detect'}")
        
        try:
            transcriber = WhisperTranscriber(
                model_size=self.whisper_model,
                device=self.device,
                language=self.language,
                verbose=True,
                temperature=0.0,     # Greedy decoding for consistency
                beam_size=5,         # Good balance of quality and speed
                best_of=5,           # Multiple candidates for better quality
                patience=1.0         # Standard beam search patience
            )

            # Transcribe with word-level timestamps for better alignment
            transcript = transcriber.transcribe(
                wav_path,
                word_timestamps=True
            )
            
            segment_count = len(transcript.get("segments", []))
            self.logger.info(f"Transcription complete with {segment_count} segments")
            
            # Get some basic stats about the transcript
            if segment_count > 0:
                total_duration = transcript["segments"][-1]["end"]
                total_words = len(transcript["text"].split())
                self.logger.info(f"Total duration: {total_duration:.2f}s, Word count: {total_words}")
            
            return transcript
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def _diarize(self, wav_path: str) -> List[Dict[str, Any]]:
        """
        Args:
          wav_path – same .wav file
        Returns:
          list of diarization segments, each with:
            - speaker: str
            - start: float (secs)
            - end: float   (secs)
        """
        self.logger.info("Step 3: Diarizing speakers")
        
        try:
            diarizer = SpeakerDiarizer(device=self.device)
            diarization = diarizer.diarize(wav_path)
            
            # Get some basic stats about the diarization
            speaker_count = len(set(s["speaker"] for s in diarization))
            segment_count = len(diarization)
            
            self.logger.info(f"Diarization complete with {speaker_count} speakers, {segment_count} segments")
            
            return diarization
        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise RuntimeError(f"Diarization failed: {str(e)}")

    def _merge(self, transcript: Dict[str, Any], diarization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        self.logger.info("Step 4: Merging transcript with speaker diarization")
        
        try:
            merger = SegmentMerger(
                max_gap=0.5,     # Half-second tolerance for non-overlapping segments
                min_overlap=0.1  # 100ms minimum overlap to associate speech with speaker
            )
            
            merged = merger.merge(
                transcript_segments=transcript["segments"],
                diarization_segments=diarization
            )
            
            # Get some insights about the merged data
            speaker_stats = {}
            for seg in merged:
                speaker = seg.get("speaker", "UNKNOWN")
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"count": 0, "duration": 0.0}
                speaker_stats[speaker]["count"] += 1
                speaker_stats[speaker]["duration"] += (seg["end"] - seg["start"])
            
            self.logger.info(f"Merge complete with {len(merged)} segments")
            for speaker, stats in speaker_stats.items():
                self.logger.info(f"  {speaker}: {stats['count']} segments, {stats['duration']:.1f}s")
            
            return merged
        except Exception as e:
            self.logger.error(f"Merge failed: {str(e)}")
            raise RuntimeError(f"Merge failed: {str(e)}")

    def _export(self, merged: List[Dict[str, Any]]) -> None:
        """
        Writes:
          - results/transcript.json (raw merged list)
          - results/transcript.md   (Markdown with headers per minute)
        """
        self.logger.info("Step 5: Exporting results to Markdown and JSON")
        
        try:
            exporter = MarkdownExporter(
                output_md=self.transcript_md,
                output_json=self.transcript_json
            )
            
            # Export JSON first (raw data)
            exporter.export_json(merged)
            
            # Export Markdown (formatted for humans)
            exporter.export_markdown(merged, block_minutes=1)
            
            self.logger.info(f"Export complete: {self.transcript_json} and {self.transcript_md}")
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise RuntimeError(f"Export failed: {str(e)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MeetingScribe: Convert meeting videos to speaker-labelled transcripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video_path",
        help="Path to input video file (.mp4, .mkv, .mov)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="results/",
        help="Output folder for transcript files"
    )
    
    parser.add_argument(
        "--lang",
        default=None,
        help="ISO-639-1 language code (None=auto-detect)"
    )
    
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small for better quality)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (auto=detect GPU automatically)"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        scribe = MeetingScribe(
            video_path=args.video_path,
            output_folder=args.output,
            language=args.lang,
            whisper_model=args.model,
            device=args.device
        )
        
        scribe.run()
        return 0  # Success exit code
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 2  # File not found exit code
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 3  # Invalid input exit code
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1  # General error exit code


if __name__ == "__main__":
    sys.exit(main())