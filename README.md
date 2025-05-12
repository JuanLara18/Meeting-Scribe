# MeetingScribe

> **One‑line pitch**: *Drop a meeting video in your terminal and get back a clean, speaker‑labelled transcript plus Markdown notes — all with free, open‑source models you can run locally in under an hour.*

---

## Features (v0.1)

| Feature | Status | Notes |
|---------|--------|-------|
| High‑quality multilingual transcription (OpenAI Whisper) | ✅ | Runs on CPU or GPU |
| Automatic speaker diarization (pyannote.audio) | ✅ | Distinguishes *who* spoke |
| Markdown export with timestamps | ✅ | Saves to `results/transcript.md` |
| One‑command CLI (`python main.py <video>`) | ✅ | Creates a `results/` folder |
| Key‑frame extraction for slide changes (LMSKE) | ⏳ | Planned v0.2 |
| Topic summarisation & action‑items (LLM) | ⏳ | Planned v0.3 |

---

## Quick Start (⏱️ ~15 min)

> **Prereqs**: Python 3.10+, `ffmpeg` (≥ 4.2).

```bash
# 1. Clone & enter
git clone https://github.com/your‑user/meetingscribe.git
cd meetingscribe

# 2. Create env & install deps
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run on a sample video
python main.py path/to/meeting.mp4
````

Output:

```
results/
├── transcript.md        # speaker‑segmented Markdown
└── transcript.json      # raw structured data
```

---

## Usage

```bash
usage: python main.py [-h] [--lang en] [--model base] VIDEO_PATH
```

* **`VIDEO_PATH`** – any local `.mp4 /.mkv /.mov` file.
* **`--lang`** – ISO‑639‑1 code for forced language (default: auto‑detect).
* **`--model`** – whisper size (`tiny` `base` `small` `medium` `large-v3`).

Example with Spanish audio, medium model:

```bash
python main.py reunión.mp4 --lang es --model medium
```

---

## Project Structure

```
meetingscribe
├── main.py              # orchestrates the end‑to‑end pipeline
├── processing/
│   ├── audio.py         # audio extraction (ffmpeg)
│   ├── transcribe.py    # whisper wrapper
│   ├── diarize.py       # pyannote wrapper
│   └── merge.py         # align speakers + text
├── utils/
│   └── markdown.py      # export helpers
├── requirements.txt
├── README.md
└── results/             # auto‑created
```

Minimal today — but every component lives in its own module so we can swap models or add GPU acceleration without touching `main.py`.

---

## Technology Stack

| Layer             | Tool                                                         | Why                                    |
| ----------------- | ------------------------------------------------------------ | -------------------------------------- |
| **ASR**           | [OpenAI Whisper](https://github.com/openai/whisper)          | State‑of‑the‑art, MIT license, offline |
| **Diarization**   | [pyannote.audio](https://github.com/pyannote/pyannote-audio) | SOTA pretrained pipelines              |
| **Media**         | [`ffmpeg`](https://ffmpeg.org/)                              | battle‑tested extraction               |
| **Future vision** | LMSKE (key‑frames), Llama‑3 local LLMs                       | keep everything free & private         |

---

## How it Works (Flow Diagram)

```
           video.mp4
                │
      ┌─────────▼─────────┐
      │ 1. ffmpeg extract │──► audio.wav
      └─────────┬─────────┘
                │
  ┌─────────────▼─────────────┐
  │ 2. Whisper ASR → segments │
  └─────────────┬─────────────┘
                │
  ┌─────────────▼─────────────┐
  │ 3. pyannote diarize audio │
  └─────────────┬─────────────┘
                │
  ┌─────────────▼─────────────┐
  │   4. Merge text+speakers   │
  └─────────────┬─────────────┘
                │
  ┌─────────────▼─────────────┐
  │ 5. Export Markdown & JSON │
  └───────────────────────────┘
```

---

## Roadmap

1. **v0.2** – Key‑frame extraction ➜ embed screenshots in Markdown.
2. **v0.3** – Local LLM summariser ➜ bullet goals, action items.
3. **v1.0** – Real‑time streaming mode & simple web UI (FastAPI + React).

---

## License

MIT — free for personal & commercial projects. Attribution welcome but not required.
