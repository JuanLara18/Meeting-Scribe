#!/usr/bin/env python
# test_components.py - Test individual components of MeetingScribe
# ───────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def test_audio_extraction(video_path, output_dir):
    """Test audio extraction component."""
    logger.info("Testing audio extraction...")
    from processing.audio import AudioExtractor
    
    extractor = AudioExtractor()
    output_wav = os.path.join(output_dir, "test_audio.wav")
    
    try:
        wav_path = extractor.extract(
            video_path=video_path,
            target_wav=output_wav,
            sample_rate=16000,
            mono=True
        )
        logger.info(f"Audio extraction successful: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}")
        return None

def test_transcription(wav_path, model_size="tiny"):
    """Test transcription component."""
    logger.info(f"Testing transcription with model '{model_size}'...")
    from processing.transcribe import WhisperTranscriber
    
    try:
        transcriber = WhisperTranscriber(
            model_size=model_size,
            verbose=True
        )
        
        transcript = transcriber.transcribe(wav_path)
        
        segment_count = len(transcript.get("segments", []))
        logger.info(f"Transcription successful with {segment_count} segments")
        
        if segment_count > 0:
            total_duration = transcript["segments"][-1]["end"]
            total_words = len(transcript["text"].split())
            logger.info(f"Duration: {total_duration:.2f}s, Words: {total_words}")
        
        return transcript
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return None

def test_diarization(wav_path):
    """Test diarization component."""
    logger.info("Testing speaker diarization...")
    
    try:
        # First try with the fixed version if available
        try:
            from processing.diarize import SpeakerDiarizer
            logger.info("Using fixed diarization module")
        except ImportError:
            from processing.diarize import SpeakerDiarizer
            logger.info("Using original diarization module")
        
        diarizer = SpeakerDiarizer(device="cpu")
        diarization = diarizer.diarize(wav_path)
        
        speaker_count = len(set(s["speaker"] for s in diarization))
        segment_count = len(diarization)
        
        logger.info(f"Diarization successful with {speaker_count} speakers, {segment_count} segments")
        return diarization
    except Exception as e:
        logger.error(f"Diarization failed: {str(e)}")
        
        # Check if it's the affinity parameter error and attempt to fix
        if "affinity" in str(e):
            logger.info("Detected AgglomerativeClustering parameter issue")
            logger.info("This is a known issue with some scikit-learn versions")
            logger.info("Use the fixed diarize.py module to resolve this")
        
        return None

def test_merger(transcript, diarization):
    """Test segment merger component."""
    logger.info("Testing segment merger...")
    from processing.merge import SegmentMerger
    
    if transcript is None or diarization is None:
        logger.error("Cannot test merger without transcript and diarization")
        return None
    
    try:
        merger = SegmentMerger()
        merged = merger.merge(
            transcript_segments=transcript["segments"],
            diarization_segments=diarization
        )
        
        logger.info(f"Merger successful with {len(merged)} segments")
        
        # Get some insights about the merged data
        speaker_stats = {}
        for seg in merged:
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"count": 0, "duration": 0.0}
            speaker_stats[speaker]["count"] += 1
            speaker_stats[speaker]["duration"] += (seg["end"] - seg["start"])
        
        for speaker, stats in speaker_stats.items():
            logger.info(f"  {speaker}: {stats['count']} segments, {stats['duration']:.1f}s")
        
        return merged
    except Exception as e:
        logger.error(f"Merger failed: {str(e)}")
        return None

def test_exporter(merged, output_dir):
    """Test Markdown exporter component."""
    logger.info("Testing Markdown exporter...")
    from utils.markdown import MarkdownExporter
    
    if merged is None:
        logger.error("Cannot test exporter without merged segments")
        return False
    
    try:
        md_path = os.path.join(output_dir, "test_transcript.md")
        json_path = os.path.join(output_dir, "test_transcript.json")
        
        exporter = MarkdownExporter(
            output_md=md_path,
            output_json=json_path
        )
        
        # Export JSON
        exporter.export_json(merged)
        
        # Export Markdown
        exporter.export_markdown(merged, block_minutes=1)
        
        logger.info(f"Export successful: {json_path} and {md_path}")
        return True
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return False

def fix_diarize_py():
    """Apply the fix to diarize.py by copying diarize.py to diarize.py."""
    logger.info("Applying fix to diarize.py...")
    
    try:
        fixed_path = Path("processing/diarize.py")
        original_path = Path("processing/diarize.py")
        backup_path = original_path.with_suffix('.py.bak')
        
        if not fixed_path.exists():
            logger.error(f"Fixed module not found: {fixed_path}")
            logger.info("Please create diarize.py first using the provided code.")
            return False
        
        # Create backup of original file
        if original_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(original_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Copy fixed version to replace original
        import shutil
        shutil.copy2(fixed_path, original_path)
        logger.info(f"Successfully replaced {original_path} with fixed version")
        
        logger.info("Fix applied successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to apply fix: {str(e)}")
        return False

def test_all_components(video_path, output_dir, model_size="tiny"):
    """Run a test of all components in sequence."""
    logger.info(f"Testing all components with video: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Audio extraction
    wav_path = test_audio_extraction(video_path, output_dir)
    if not wav_path:
        logger.error("Audio extraction failed, cannot continue")
        return False
    
    # Step 2: Transcription
    transcript = test_transcription(wav_path, model_size)
    if not transcript:
        logger.error("Transcription failed, cannot continue")
        return False
    
    # Step 3: Diarization
    diarization = test_diarization(wav_path)
    if not diarization:
        logger.error("Diarization failed, cannot continue")
        return False
    
    # Step 4: Segment merger
    merged = test_merger(transcript, diarization)
    if not merged:
        logger.error("Merger failed, cannot continue")
        return False
    
    # Step 5: Export
    if not test_exporter(merged, output_dir):
        logger.error("Export failed")
        return False
    
    logger.info("All components tested successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Test individual components of MeetingScribe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--video", "-v",
        help="Path to test video file (for full pipeline test)"
    )
    
    parser.add_argument(
        "--audio", "-a",
        help="Path to existing audio file (to skip extraction)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="test_results",
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for transcription test"
    )
    
    parser.add_argument(
        "--test-extraction",
        action="store_true",
        help="Test audio extraction"
    )
    
    parser.add_argument(
        "--test-transcription",
        action="store_true",
        help="Test transcription"
    )
    
    parser.add_argument(
        "--test-diarization",
        action="store_true",
        help="Test diarization"
    )
    
    parser.add_argument(
        "--test-merger",
        action="store_true",
        help="Test segment merger"
    )
    
    parser.add_argument(
        "--test-exporter",
        action="store_true",
        help="Test Markdown exporter"
    )
    
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all components in sequence"
    )
    
    parser.add_argument(
        "--apply-fix",
        action="store_true",
        help="Apply the diarization fix"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Apply fix if requested
    if args.apply_fix:
        if fix_diarize_py():
            return 0
        else:
            return 1
    
    # Test all components
    if args.test_all:
        if args.video:
            if test_all_components(args.video, args.output, args.model):
                return 0
            else:
                return 1
        else:
            logger.error("Video path required for full test")
            return 1
    
    # Track which tests ran and their results
    test_results = {}
    
    # Audio path for tests
    audio_path = args.audio
    transcript = None
    diarization = None
    merged = None
    
    # Test audio extraction if requested
    if args.test_extraction:
        if args.video:
            audio_path = test_audio_extraction(args.video, args.output)
            test_results["audio_extraction"] = audio_path is not None
        else:
            logger.error("Video path required for extraction test")
            test_results["audio_extraction"] = False
    
    # Test transcription if requested
    if args.test_transcription:
        if audio_path:
            transcript = test_transcription(audio_path, args.model)
            test_results["transcription"] = transcript is not None
        else:
            logger.error("Audio path required for transcription test")
            test_results["transcription"] = False
    
    # Test diarization if requested
    if args.test_diarization:
        if audio_path:
            diarization = test_diarization(audio_path)
            test_results["diarization"] = diarization is not None
        else:
            logger.error("Audio path required for diarization test")
            test_results["diarization"] = False
    
    # Test merger if requested
    if args.test_merger:
        if transcript is None and audio_path and not args.test_transcription:
            logger.info("Transcribing for merger test...")
            transcript = test_transcription(audio_path, args.model)
        
        if diarization is None and audio_path and not args.test_diarization:
            logger.info("Diarizing for merger test...")
            diarization = test_diarization(audio_path)
        
        merged = test_merger(transcript, diarization)
        test_results["merger"] = merged is not None
    
    # Test exporter if requested
    if args.test_exporter:
        if merged is None and transcript is not None and diarization is not None:
            logger.info("Merging for exporter test...")
            from processing.merge import SegmentMerger
            merger = SegmentMerger()
            merged = merger.merge(transcript["segments"], diarization)
        
        if merged:
            exporter_result = test_exporter(merged, args.output)
            test_results["exporter"] = exporter_result
        else:
            logger.error("Merged segments required for exporter test")
            test_results["exporter"] = False
    
    # Print summary of results
    if test_results:
        logger.info("\n=== Test Results Summary ===")
        for test, result in test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test.replace('_', ' ').title()}: {status}")
        
        # Overall result
        if all(test_results.values()):
            logger.info("\nAll tests passed successfully!")
            return 0
        else:
            logger.error("\nSome tests failed. See above for details.")
            return 1
    else:
        logger.info("\nNo tests were run. Use --help to see available options.")
        return 0

if __name__ == "__main__":
    sys.exit(main())