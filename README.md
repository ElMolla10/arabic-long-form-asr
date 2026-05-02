# Arabic Long-Form ASR Improvement Pipeline

This project focuses on improving Arabic automatic speech recognition (ASR) for long-duration audio such as lectures, podcasts, khutbahs, interviews, and archived educational media.

Modern Whisper-based ASR systems work best on short windows of audio. This project builds a robust long-form transcription pipeline that chunks long recordings, preserves context between chunks, reconstructs the final transcript, normalizes Arabic text, and evaluates transcription quality using WER and CER.

The clip generation script is included as an optional downstream demo, but the main project evaluation is based on transcription quality.

## Final Results Summary

The project now includes both pipeline-level long-form experiments and model-level fine-tuning experiments.

### 30+ Minute Long-Form Benchmark

- Dataset: Google FLEURS Arabic Egypt validation split.
- Construction: 167 validation clips concatenated into one long WAV file.
- Duration: 1804.44 seconds = 30.07 minutes.
- Reference: official FLEURS transcripts concatenated in the same order.
- Reference size: 3,035 words / 17,717 characters.
- ASR model for local long-form tests: whisper.cpp `ggml-base.bin`.

| System | Chunk Size | Overlap | Context | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| `whisper_base_30min_no_context` | 60s | 2s | No | 0.5414 | **0.4926** | **0.2069** |
| `whisper_base_30min_context` | 60s | 2s | 24 words | 0.6250 | 0.5645 | 0.3259 |
| `whisper_base_30min_no_context_30s` | 30s | 2s | No | 0.5621 | 0.5153 | 0.2402 |

Best long-form setting: **60-second chunks, 2-second overlap, no prompt context**.

### Fine-Tuning Experiments

Fine-tuning experiments were run on Kaggle GPU with FLEURS Arabic Egypt. Evaluation used 20 held-out validation samples.

| Experiment | Train Samples | Augmentation | Eval Loss | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| E4 Small fine-tune | 50 | No | 2.0174 | 0.2899 | 0.2671 | 0.0675 |
| E5 Larger fine-tune | 150 | No | **1.6712** | **0.2769** | **0.2508** | **0.0618** |
| E6 Larger fine-tune + augmentation | 150 | Yes | 1.6823 | **0.2769** | **0.2508** | **0.0618** |

Best model-level result: **E5 larger fine-tune**. Increasing training data from 50 to 150 samples improved WER/CER. The augmentation configuration used in E6 did not improve WER/CER over E5.

### Key Findings

- The pipeline successfully handles a 30+ minute Arabic ASR benchmark.
- Arabic normalization reduced measured WER for every long-form system.
- Prompt context did not help in these experiments; it likely propagated recognition errors.
- 60-second chunks performed better than 30-second chunks for the long-form benchmark.
- Fine-tuning improved Arabic ASR quality on held-out FLEURS validation samples.
- MALIK/clipper is a downstream bonus component; the main evaluation is ASR quality.

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
| `Whisper_Fine_Tuning_Experiments_E4_E6.ipynb` | Colab-ready notebook for fine-tuning experiments E4-E6 |
| `asr_experiments.py` | Main ASR experiment pipeline: manifest creation, long-form transcription, Arabic normalization, WER/CER evaluation |
| `clipper.py` | Optional bonus component for creating captioned media clips from transcripts |
| `README_ASR_IMPROVEMENT.md` | Detailed ASR experiment plan and requirement mapping |
| `requirements.txt` | Python dependencies |
| `DEMO.md` | Presentation demo steps |
| `RESULTS_DEMO.md` | Actual local demo results from a public Arabic FLEURS sample |
| `RESULTS_LONGFORM_30MIN.md` | Actual 30+ minute long-form evaluation results |
| `Arabic_LongForm_ASR_Pipeline.pptx` | Presentation deck, if included in the repository |

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

Completed experiments:

| Experiment | Description | Status |
|---|---|---|
| E0 | Whisper baseline / long-form benchmark | Completed |
| E1 | 60s chunking without prompt context | Completed |
| E2 | 60s chunking with 24-word prompt context | Completed |
| E3 | 30s vs 60s chunk-size comparison | Completed |
| E4 | Small fine-tuned model, 50 samples | Completed |
| E5 | Larger fine-tuned model, 150 samples | Completed |
| E6 | Larger fine-tuned model with augmentation | Completed |
| E7 | Arabic normalization on/off comparison | Completed |

WER and CER are reported in `RESULTS_LONGFORM_30MIN.md`.

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

## Optional Bonus: Clip Generation

`clipper.py` uses transcripts to create short captioned clips from long media files. This is useful for educational media reuse and social-media publishing, but it is not the main ASR evaluation component.
