# CLAUDE.md - AI Assistant Guide for MeetingScribe

This document provides comprehensive guidance for AI assistants (like Claude) working on the MeetingScribe codebase. It covers project structure, architecture, development workflows, conventions, and best practices.

---

## Project Overview

**MeetingScribe** is an open-source Python application that converts meeting videos into speaker-labelled transcripts using local, privacy-preserving AI models.

**Key Features:**
- High-quality multilingual transcription (OpenAI Whisper)
- Automatic speaker diarization (custom implementation)
- Markdown export with timestamps
- One-command CLI interface
- Fully offline, no cloud dependencies

**Use Case:** Drop a meeting video (`.mp4`, `.mkv`, `.mov`) into the CLI and receive a clean, speaker-segmented transcript in Markdown format.

---

## Repository Structure

```
Meeting-Scribe/
├── main.py                      # Pipeline orchestrator (entry point)
├── processing/                  # Core processing modules
│   ├── audio.py                # Audio extraction via FFmpeg
│   ├── transcribe.py           # Whisper ASR wrapper
│   ├── diarize.py              # Custom speaker diarization
│   └── merge.py                # Alignment of transcript + speakers
├── utils/
│   └── markdown.py             # Markdown/JSON export utilities
├── scripts/                     # Setup and testing scripts
│   ├── setup.py                # Automated installation script
│   ├── whisper_install.py      # Python 3.13 Whisper installer
│   └── test_components.py      # Component testing suite
├── config/
│   └── requirements.txt        # Python dependencies
├── docs/
│   └── INSTALL.md              # Installation guide
├── whisper_src/                # Local Whisper source (gitignored)
├── results/                     # Output folder (gitignored)
│   ├── transcript.md           # Human-readable transcript
│   ├── transcript.json         # Raw structured data
│   └── audio.wav               # Extracted audio
├── README.md                    # User-facing documentation
├── .gitignore                   # Git exclusions
└── CLAUDE.md                    # This file
```

### Key Files Explanation

| File | Purpose | When to Modify |
|------|---------|----------------|
| `main.py` | Pipeline orchestrator; coordinates all processing steps | When adding new pipeline stages or CLI options |
| `processing/audio.py` | FFmpeg wrapper for video→audio extraction | When changing audio preprocessing (sample rate, channels) |
| `processing/transcribe.py` | Whisper ASR interface | When updating Whisper models or adding transcription options |
| `processing/diarize.py` | Custom speaker diarization using MFCC+clustering | When improving speaker detection algorithms |
| `processing/merge.py` | Aligns transcript segments with speaker turns | When adjusting overlap/gap tolerances |
| `utils/markdown.py` | Exports results to Markdown/JSON | When changing output format or adding metadata |

---

## Architecture & Pipeline

### Processing Pipeline (5 Steps)

```
video.mp4
    │
    ▼
┌─────────────────────┐
│ 1. Audio Extraction │  → audio.py: ffmpeg → 16kHz mono WAV
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Transcription    │  → transcribe.py: Whisper → timestamped text segments
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Diarization      │  → diarize.py: MFCC features + clustering → speaker turns
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Merge            │  → merge.py: align text + speakers by timestamp overlap
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Export           │  → markdown.py: write transcript.md and transcript.json
└─────────────────────┘
```

### Component Responsibilities

#### 1. `AudioExtractor` (audio.py)
- **Input:** Video file path
- **Output:** 16kHz mono WAV audio
- **Dependencies:** FFmpeg binary
- **Key Method:** `extract(video_path, target_wav, sample_rate=16000, mono=True)`
- **Error Handling:** Checks FFmpeg availability, validates input file exists

#### 2. `WhisperTranscriber` (transcribe.py)
- **Input:** WAV file path
- **Output:** Dict with `text` (full transcript) and `segments` (timestamped chunks)
- **Models:** tiny, base, small, medium, large-v3
- **Device:** CPU or CUDA
- **Key Method:** `transcribe(wav_path, output_json=False)`
- **Lazy Loading:** Model loads on first use to save memory

#### 3. `SpeakerDiarizer` (diarize.py)
- **Input:** WAV file path
- **Output:** List of speaker segments: `[{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.75}, ...]`
- **Algorithm:**
  1. Energy-based Voice Activity Detection (VAD)
  2. MFCC feature extraction (20 coefficients + deltas)
  3. Agglomerative clustering (Ward linkage)
  4. Elbow method for optimal speaker count
  5. Post-processing (merge adjacent segments)
- **Key Method:** `diarize(wav_path, min_speakers=None, max_speakers=None)`
- **Note:** Custom implementation replaces pyannote.audio to avoid Hugging Face dependencies

#### 4. `SegmentMerger` (merge.py)
- **Input:** Transcript segments + diarization segments
- **Output:** Unified segments with speaker labels + text
- **Algorithm:**
  - Overlap-based matching (min_overlap threshold)
  - Proximity-based fallback for non-overlapping segments
  - Context-based assignment for unmatched segments
- **Parameters:**
  - `max_gap`: 0.5s (max distance to still associate text→speaker)
  - `min_overlap`: 0.1s (min overlap required for match)
- **Key Method:** `merge(transcript_segments, diarization_segments)`

#### 5. `MarkdownExporter` (markdown.py)
- **Input:** Merged segments
- **Output:**
  - `transcript.json`: Raw JSON dump
  - `transcript.md`: Formatted Markdown with minute-based headers
- **Format:** Speaker names in bold, timestamps in code blocks, grouped by time
- **Key Methods:** `export_json()`, `export_markdown(block_minutes=1)`

---

## Development Workflows

### Setting Up the Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JuanLara18/Meeting-Scribe.git
   cd Meeting-Scribe
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r config/requirements.txt
   ```

4. **For Python 3.13 users (Whisper fix):**
   ```bash
   python scripts/whisper_install.py
   ```

5. **Verify installation:**
   ```bash
   python main.py --help
   ```

### Running the Application

**Basic usage:**
```bash
python main.py path/to/video.mp4
```

**With options:**
```bash
python main.py meeting.mp4 --lang es --model medium --output results/
```

**CLI Arguments:**
- `video_path` (required): Path to input video
- `--output`, `-o`: Output folder (default: `results/`)
- `--lang`: ISO-639-1 language code (default: auto-detect)
- `--model`: Whisper model size (default: `base`)
- `--verbose`, `-v`: Enable debug logging

### Testing Components

Run the test suite:
```bash
python scripts/test_components.py
```

This tests:
- Audio extraction
- Transcription
- Diarization
- Merging
- Markdown export

---

## Code Conventions & Best Practices

### Python Style
- **Version:** Compatible with Python 3.10 - 3.13
- **Style Guide:** Follow PEP 8
- **Imports:** Group in order: standard library → third-party → local modules
- **Type Hints:** Use where it improves clarity (already used in function signatures)
- **Docstrings:** Google-style docstrings for all public methods

### Logging
- **Framework:** Python's built-in `logging` module
- **Levels:**
  - `INFO`: Pipeline progress, major steps
  - `DEBUG`: Detailed diagnostics (enable with `--verbose`)
  - `WARNING`: Recoverable issues (e.g., unmatched segments)
  - `ERROR`: Failures requiring user attention
- **Format:** `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

**Example:**
```python
self.logger.info("Step 1: Extracting audio from video")
self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
self.logger.warning(f"{unknown_count} segments have UNKNOWN speaker")
self.logger.error(f"Transcription failed: {str(e)}")
```

### Error Handling
- **Validation:** Check inputs early (file existence, parameter ranges)
- **Exceptions:** Raise specific exceptions (`FileNotFoundError`, `ValueError`, `RuntimeError`)
- **Context:** Include helpful error messages with context
- **Cleanup:** No temporary files to clean (audio.wav preserved for debugging)

**Example:**
```python
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

if whisper_model not in valid_models:
    raise ValueError(f"Invalid model: {whisper_model}. Must be one of {valid_models}")
```

### File Organization
- **One class per file** (current pattern)
- **Modules are self-contained:** Each processing module has minimal external dependencies
- **Lazy loading:** Models load on first use (see `WhisperTranscriber._load_model()`)
- **No circular imports:** Clean dependency graph (utils ← processing ← main)

### Naming Conventions
| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `WhisperTranscriber` |
| Functions/Methods | snake_case | `extract_audio()` |
| Constants | UPPER_SNAKE_CASE | `VALID_MODELS` |
| Private methods | _leading_underscore | `_load_model()` |
| Module files | snake_case | `markdown.py` |

---

## Key Technical Details

### Dependencies
**Core:**
- `ffmpeg-python`, `moviepy`, `pydub`: Media processing
- `torch`, `torchaudio`: Deep learning framework
- `whisper`: OpenAI Whisper ASR (installed from local source)
- `librosa`, `soundfile`: Audio analysis
- `scikit-learn`: Clustering algorithms
- `numpy`, `scipy`: Numerical computing

**Optional:**
- CUDA-enabled PyTorch for GPU acceleration

### Audio Processing Specs
- **Sample Rate:** 16 kHz (optimal for Whisper)
- **Channels:** Mono (required for diarization)
- **Format:** PCM 16-bit WAV
- **Frame Length:** 25ms (diarization)
- **Frame Shift:** 10ms (diarization)

### Whisper Models
| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| tiny | 39M | Fastest | Low | Quick tests |
| base | 74M | Fast | Good | Default |
| small | 244M | Medium | Better | Balanced |
| medium | 769M | Slow | High | Accuracy focus |
| large-v3 | 1550M | Slowest | Highest | Production |

### Diarization Algorithm Details
**Feature Extraction:**
- 20 MFCC coefficients
- Delta coefficients (velocity)
- Delta-delta coefficients (acceleration)
- Total: 60 features per frame

**VAD (Voice Activity Detection):**
- Energy-based thresholding
- Threshold: 0.3 (tunable via `self.vad_threshold`)
- Padding: 0.1s before/after speech

**Clustering:**
- Algorithm: Agglomerative (Ward linkage)
- Distance: Euclidean
- Speaker estimation: Elbow method on distortion curve

---

## Common Modification Scenarios

### Adding a New Whisper Model
1. Update `valid_models` list in `main.py:74` and `transcribe.py:51`
2. No other changes needed (Whisper handles model loading)

### Changing Audio Sample Rate
1. Modify `sample_rate` in `main.py:120` (default: 16000)
2. **Caution:** Whisper is optimized for 16kHz; other rates may reduce accuracy

### Adjusting Speaker Merging Tolerance
1. Edit `max_gap` in `main.py:209` (default: 0.5s)
2. Edit `min_overlap` in `main.py:210` (default: 0.1s)
3. **Effect:** Higher values = more aggressive merging

### Changing Markdown Output Format
1. Modify `MarkdownExporter.export_markdown()` in `utils/markdown.py:91`
2. Current format: Minute-based headers, bold speaker names, code-block timestamps
3. Example change: Add speaker color coding, conversation threads, etc.

### Supporting Additional Video Formats
1. No code changes needed (FFmpeg handles most formats)
2. If issues arise, check FFmpeg codec support with: `ffmpeg -formats`

### Adding GPU Support
1. Change `device="cpu"` to `device="cuda"` in `main.py:176`
2. Ensure CUDA-enabled PyTorch is installed (see `requirements.txt` comments)

---

## Testing Guidelines

### Unit Testing (Future)
- **Framework:** pytest (not yet implemented)
- **Target Coverage:** Core processing modules
- **Mock Data:** Use short test audio clips (<10s)

### Integration Testing
- Use `scripts/test_components.py` for end-to-end validation
- Provides sample data and checks each pipeline stage

### Manual Testing Checklist
- [ ] Audio extraction completes without errors
- [ ] Transcription produces sensible text
- [ ] Diarization detects correct number of speakers
- [ ] Merged segments have speaker labels
- [ ] Markdown output is readable and properly formatted
- [ ] JSON output is valid and complete

---

## Known Issues & Limitations

### Current Limitations
1. **No real-time processing:** Batch-only (entire video must be processed)
2. **Limited speaker identification:** Labels are generic (SPEAKER_00, SPEAKER_01)
3. **No video analysis:** Key-frame extraction planned for v0.2
4. **Single-threaded:** No parallel processing (future optimization)
5. **Memory usage:** Large videos may require significant RAM

### Python 3.13 Compatibility
- **Issue:** Whisper's `pyproject.toml` has dynamic versioning incompatible with Python 3.13
- **Solution:** Use `scripts/whisper_install.py` to patch and install from source
- **Details:** Script modifies version settings and installs with `--no-build-isolation`

### Speaker Diarization Accuracy
- **Algorithm:** Custom implementation (not production-grade)
- **Limitations:**
  - May struggle with overlapping speech
  - Requires distinct speaker voices
  - Optimal for 2-5 speakers
- **Future:** Consider pyannote.audio integration with optional Hugging Face auth

---

## Extending the Project

### Planned Features (Roadmap)
1. **v0.2:** Key-frame extraction (embed screenshots in Markdown)
2. **v0.3:** Local LLM summarization (action items, bullet points)
3. **v1.0:** Real-time streaming mode + web UI (FastAPI + React)

### Adding a New Processing Stage
**Example: Topic Detection**

1. **Create new module:** `processing/topics.py`
   ```python
   class TopicDetector:
       def detect(self, merged_segments):
           # Your logic here
           return topic_segments
   ```

2. **Update main.py pipeline:**
   ```python
   # In MeetingScribe.run():
   topics = self._detect_topics(merged)
   ```

3. **Add export support in markdown.py:**
   ```python
   def export_with_topics(self, merged, topics):
       # Format topics in Markdown
   ```

### Adding New Output Formats
1. Create new exporter in `utils/` (e.g., `utils/pdf.py`, `utils/srt.py`)
2. Add CLI flag in `main.py` (e.g., `--format pdf`)
3. Call exporter in `MeetingScribe._export()`

---

## Debugging Tips

### Common Issues

**"ffmpeg not found":**
- Verify installation: `ffmpeg -version`
- Check PATH environment variable
- On Windows, may need terminal restart after install

**"Whisper module not found":**
- Ensure virtual environment is activated
- For Python 3.13: Run `python scripts/whisper_install.py`
- Otherwise: `pip install git+https://github.com/openai/whisper.git`

**"No speech detected in audio":**
- Check VAD threshold in `diarize.py:45` (try lowering to 0.2)
- Verify audio file has actual speech (play in media player)
- Check audio extraction succeeded (inspect `results/audio.wav`)

**"UNKNOWN speaker labels":**
- Indicates merge algorithm couldn't match text to speaker
- Adjust `max_gap` and `min_overlap` in `merge.py` or `main.py:209-210`
- Review diarization output (may have missed speech segments)

### Logging Debug Output
Enable verbose mode to see detailed logs:
```bash
python main.py video.mp4 --verbose
```

This shows:
- FFmpeg commands
- Model loading times
- Segment counts
- Speaker statistics
- Feature extraction details

### Inspecting Intermediate Outputs
Check `results/` folder:
- `audio.wav`: Verify audio extraction quality
- `transcript.json`: Raw data for debugging merging issues

---

## Git Workflow

### Branch Strategy
- **Main branch:** `main` (stable releases)
- **Feature branches:** `feature/feature-name` or `claude/session-id`
- **Development:** Work on feature branches, PR to main

### Commit Messages
Follow conventional commits:
- `feat: add key-frame extraction`
- `fix: resolve speaker overlap issue`
- `docs: update installation guide`
- `refactor: simplify merge algorithm`
- `test: add diarization unit tests`

### Ignored Files (`.gitignore`)
- `__pycache__/`, `*.pyc`: Python bytecode
- `.venv/`, `venv/`: Virtual environments
- `results/`: Output folder (user-generated)
- `*.mp4`, `*.mkv`: Video files (too large)
- `whisper_src/`: Local Whisper source
- `.env`: Environment variables (secrets)
- `.vscode/`, `.idea/`: IDE configs

---

## AI Assistant Guidelines

### When Working on This Codebase

1. **Read before writing:**
   - Always check existing implementations before adding new code
   - Understand the pipeline flow (extract → transcribe → diarize → merge → export)

2. **Maintain modularity:**
   - Keep processing modules independent
   - Avoid coupling between audio/transcribe/diarize components
   - Use clear interfaces (input/output contracts)

3. **Preserve error handling:**
   - Don't remove validation checks
   - Add context to error messages
   - Log at appropriate levels

4. **Document changes:**
   - Update docstrings when modifying public APIs
   - Add comments for complex algorithms (e.g., elbow method in diarize.py)
   - Update this CLAUDE.md if architecture changes

5. **Test before committing:**
   - Run `python scripts/test_components.py`
   - Test with sample video if possible
   - Verify output in `results/` folder

6. **Respect conventions:**
   - Follow existing naming patterns
   - Use logging instead of print statements
   - Keep classes focused (single responsibility)

### Quick Reference: Where to Find Things

| Task | File | Line/Method |
|------|------|-------------|
| CLI argument parsing | `main.py` | Line 262-299 |
| Pipeline orchestration | `main.py` | `MeetingScribe.run()` |
| FFmpeg command building | `processing/audio.py` | Line 79-95 |
| Whisper model loading | `processing/transcribe.py` | `_load_model()` |
| VAD algorithm | `processing/diarize.py` | `_detect_speech()` |
| Speaker clustering | `processing/diarize.py` | `_cluster_speakers()` |
| Segment overlap calculation | `processing/merge.py` | `_compute_overlap()` |
| Markdown formatting | `utils/markdown.py` | `export_markdown()` |

---

## Resources

### External Documentation
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [librosa Documentation](https://librosa.org/doc/latest/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

### Project Documentation
- `README.md`: User-facing guide
- `docs/INSTALL.md`: Detailed installation instructions
- This file (`CLAUDE.md`): Developer/AI assistant guide

### Contact & Contributing
- **Repository:** https://github.com/JuanLara18/Meeting-Scribe
- **License:** MIT
- **Issues:** GitHub Issues
- **Contributions:** PRs welcome (follow conventional commits)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2024 | Initial release with transcription + diarization |
| Current | 2025-11-14 | Added CLAUDE.md documentation |

---

*Last updated: 2025-11-14*
*Maintained by: MeetingScribe Team*
