# Arabic Long-Form ASR Improvement Pipeline

This project focuses on improving Arabic automatic speech recognition (ASR) for long-duration audio such as lectures, podcasts, khutbahs, interviews, and archived educational media.

Modern Whisper-based ASR systems work best on short windows of audio. This project builds a robust long-form transcription pipeline that chunks long recordings, preserves context between chunks, reconstructs the final transcript, normalizes Arabic text, and evaluates transcription quality using WER and CER.

The clip generation script is included as an optional downstream demo, but the main project evaluation is based on transcription quality.

## Project Objectives

- Define a clear Whisper baseline before improvement.
- Handle long-form Arabic audio of 30 minutes or more.
- Segment audio into manageable chunks.
- Preserve context continuity across chunks.
- Merge and post-process transcripts.
- Apply Arabic text normalization before evaluation.
- Compare systems using WER and CER.
- Present structured experiments with clear analysis.

## Repository Contents

| File | Purpose |
|---|---|
| `Arabic_Long_Form_ASR_Improvement.ipynb` | Notebook version of the project with logical cells for presentation and grading |
| `asr_experiments.py` | Main ASR experiment pipeline: manifest creation, long-form transcription, Arabic normalization, WER/CER evaluation |
| `clipper.py` | Optional bonus component for creating captioned media clips from transcripts |
| `README_ASR_IMPROVEMENT.md` | Detailed ASR experiment plan and requirement mapping |
| `requirements.txt` | Python dependencies |
| `DEMO.md` | Presentation demo steps |

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

System dependencies:

- `ffmpeg`
- `ffprobe`
- `whisper.cpp` with `whisper-cli`
- A Whisper model file, for example `ggml-large-v3-turbo.bin`

## Dataset / Manifest Format

Create a validation manifest as JSONL:

```json
{"id": "lecture_001", "audio": "data/audio/lecture_001.mp4", "reference": "manually corrected Arabic transcript"}
```

Each line should contain:

- `id`: unique sample ID
- `audio`: path to audio/video file
- `reference`: gold transcript for evaluation

You can also create a manifest from matching audio files and `.txt` references:

```bash
python asr_experiments.py make-manifest \
  --audio-dir data/audio \
  --ref-dir data/references \
  --audio-glob "*.mp4" \
  --out data/val.jsonl
```

## Baseline Experiment

Run Whisper before any fine-tuning or improvement:

```bash
python asr_experiments.py transcribe \
  --manifest data/val.jsonl \
  --model ~/Documents/whisper-models/ggml-large-v3-turbo.bin \
  --system-name whisper_baseline \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --context-words 24 \
  --out results/whisper_baseline_predictions.jsonl
```

Evaluate:

```bash
python asr_experiments.py evaluate \
  --pred results/whisper_baseline_predictions.jsonl \
  --out results/whisper_baseline_metrics.json \
  --csv results/whisper_baseline_metrics.csv
```

## Improved Model Experiment

Use the same evaluation manifest with the improved/fine-tuned model:

```bash
python asr_experiments.py transcribe \
  --manifest data/val.jsonl \
  --model path/to/improved-whisper-model.bin \
  --system-name whisper_improved \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --context-words 24 \
  --out results/whisper_improved_predictions.jsonl
```

Evaluate:

```bash
python asr_experiments.py evaluate \
  --pred results/whisper_improved_predictions.jsonl \
  --out results/whisper_improved_metrics.json \
  --csv results/whisper_improved_metrics.csv
```

## Structured Experiments

Recommended experiments:

| Experiment | Description |
|---|---|
| E0 | Whisper baseline before fine-tuning |
| E1 | Smaller dataset fine-tuning |
| E2 | Larger dataset fine-tuning |
| E3 | Fine-tuning with augmentation |
| E4 | Arabic normalization on/off comparison |
| E5 | Long-form transcription with chunking/context reconstruction |

Report WER and CER for every experiment.

## Arabic Normalization

The evaluator normalizes Arabic before scoring:

- removes diacritics
- removes tatweel
- normalizes Alef variants to `ا`
- normalizes `ى` to `ي`
- normalizes `ة` to `ه`
- normalizes hamza forms
- removes punctuation
- collapses whitespace

This makes evaluation more focused on word recognition quality instead of spelling variants.

## Long-Form Audio Strategy

The pipeline handles long recordings by:

1. Converting input to 16 kHz mono audio.
2. Splitting audio into fixed-length chunks.
3. Adding overlap between chunks.
4. Passing previous transcript words as prompt context.
5. Transcribing each chunk.
6. Removing duplicate overlap text.
7. Reconstructing the final transcript.
8. Applying light punctuation and spacing cleanup.

## Demo

See [DEMO.md](DEMO.md) for the presentation demo flow.

## Optional Bonus: Clip Generation

`clipper.py` uses transcripts to create short captioned clips from long media files. This is useful for educational media reuse and social-media publishing, but it is not the main ASR evaluation component.
