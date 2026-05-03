# Arabic Long-Form ASR Improvement Pipeline

This project improves Arabic automatic speech recognition (ASR) for long-duration audio such as lectures, podcasts, khutbahs, interviews, educational videos, and broadcast archives.

Arabic ASR is harder than English ASR because high-quality labeled Arabic speech data is more limited, Arabic has rich morphology, and transcription can vary because of spelling forms, diacritics, dialect influence, and punctuation. This project focuses on Modern Standard Arabic (MSA) and builds a practical end-to-end pipeline using Whisper-based transcription, Arabic preprocessing, long-form chunking, adaptation experiments, and WER/CER evaluation.

## Final Deliverables

| File / Folder | Purpose |
|---|---|
| `Arabic_ASR_Final_Project_Unified.ipynb` | Main final notebook for grading and demo. |
| `Arabic_LongForm_ASR_Pipeline.pptx` | Presentation deck. |
| `FINAL_PROJECT_GUIDE.md` | Short guide for submission, demo flow, and assignment mapping. |
| `requirements.txt` | Python dependencies. |
| `asr_experiments.py` | Command-line ASR pipeline for manifests, long-form transcription, normalization, and evaluation. |
| `Clipper.py` | Optional downstream clip-generation component. |
| `data/` | Demo and 30-minute long-form JSONL manifests. |
| `results/` | Saved prediction and metric files from completed experiments. |

## Project Objectives

- Define a real-world Arabic long-form transcription problem.
- Build a complete ASR system pipeline.
- Fine-tune Whisper on Arabic data.
- Compare different dataset sizes.
- Apply data augmentation such as noise and speed perturbation.
- Apply Arabic preprocessing and normalization.
- Evaluate transcription quality using WER and CER.
- Deliver a professional presentation and working demo.

## System Pipeline

```text
Long Arabic Audio
        |
        v
Audio Preprocessing
16 kHz mono conversion, filtering, chunking
        |
        v
Whisper ASR Baseline
        |
        v
Arabic Text Normalization
        |
        v
WER / CER Evaluation
        |
        v
Adaptation Experiments
dataset size + augmentation + fine-tuning
        |
        v
Improved Whisper Model
        |
        v
Final Transcript + Results Table
```

## Final Results Summary

The repository includes completed long-form pipeline experiments and model-level fine-tuning results.

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

Best long-form setting:

```text
60-second chunks, 2-second overlap, no prompt context
```

### Fine-Tuning Experiments

Fine-tuning experiments were run on Kaggle GPU with FLEURS Arabic Egypt. Evaluation used 20 held-out validation samples.

| Experiment | Train Samples | Augmentation | Eval Loss | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| E4 Small fine-tune | 50 | No | 2.0174 | 0.2899 | 0.2671 | 0.0675 |
| E5 Larger fine-tune | 150 | No | **1.6712** | **0.2769** | **0.2508** | **0.0618** |
| E6 Larger fine-tune + augmentation | 150 | Yes | 1.6823 | **0.2769** | **0.2508** | **0.0618** |

Best model-level result:

```text
E5 larger fine-tune
```

Increasing training data from 50 to 150 samples improved WER/CER. The augmentation configuration tested in E6 did not improve over E5.

## Key Findings

- The pipeline successfully handles a 30+ minute Arabic ASR benchmark.
- Arabic normalization reduced measured WER for every long-form system.
- Prompt context did not help in these experiments; it likely propagated recognition errors.
- 60-second chunks performed better than 30-second chunks for the long-form benchmark.
- Fine-tuning improved Arabic ASR quality on held-out FLEURS validation samples.
- `Clipper.py` is an optional downstream component; the main project evaluation is ASR quality.

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

System dependencies for local transcription:

```text
ffmpeg
ffprobe
whisper.cpp with whisper-cli
a Whisper model file, such as ggml-base.bin
```

## Kaggle / Colab Setup

For FLEURS, use `datasets<4` because `google/fleurs` relies on a dataset loading script.

Run this once:

```python
!pip uninstall -y datasets huggingface_hub
!pip install -q --no-cache-dir "datasets==3.6.0" "huggingface_hub<1.0.0" transformers accelerate evaluate jiwer soundfile librosa audiomentations
```

Then restart the runtime/session and load FLEURS with:

```python
from datasets import load_dataset

dataset = load_dataset("google/fleurs", "ar_eg", trust_remote_code=True)
```

## Dataset / Manifest Format

The command-line pipeline uses JSONL manifests:

```json
{"id": "lecture_001", "audio": "data/audio/lecture_001.wav", "reference": "manually corrected Arabic transcript"}
```

Each row contains:

- `id`: unique sample ID.
- `audio`: path to audio or video file.
- `reference`: gold transcript for evaluation.

Existing manifests:

```text
data/demo.jsonl
data/longform_30min.jsonl
```

## Command-Line Usage

Create a manifest from matching audio files and reference text files:

```bash
python asr_experiments.py make-manifest \
  --audio-dir data/audio \
  --ref-dir data/references \
  --audio-glob "*.wav" \
  --out data/val.jsonl
```

Run long-form transcription with whisper.cpp:

```bash
python asr_experiments.py transcribe \
  --manifest data/val.jsonl \
  --model /path/to/ggml-base.bin \
  --system-name whisper_baseline \
  --chunk-seconds 60 \
  --overlap-seconds 2 \
  --no-prompt-context \
  --out results/whisper_baseline_predictions.jsonl
```

Evaluate predictions:

```bash
python asr_experiments.py evaluate \
  --pred results/whisper_baseline_predictions.jsonl \
  --out results/whisper_baseline_metrics.json \
  --csv results/whisper_baseline_metrics.csv
```

Evaluate one of the saved result files:

```bash
python asr_experiments.py evaluate \
  --pred results/whisper_base_30min_no_context_predictions.jsonl \
  --out results/check_metrics.json \
  --csv results/check_metrics.csv
```

## Arabic Normalization

The evaluator normalizes Arabic before scoring:

- Removes diacritics.
- Removes tatweel.
- Normalizes Alef variants to `ا`.
- Normalizes `ى` to `ي`.
- Normalizes `ة` to `ه`.
- Normalizes hamza forms.
- Converts Arabic-Indic digits.
- Removes punctuation.
- Collapses whitespace.

This makes evaluation focus more on recognition quality and less on spelling variants.

## Long-Form Audio Strategy

The pipeline handles long recordings by:

1. Converting input to 16 kHz mono audio.
2. Splitting audio into fixed-length chunks.
3. Adding overlap between chunks.
4. Optionally passing previous transcript words as prompt context.
5. Transcribing each chunk.
6. Removing duplicate overlap text.
7. Reconstructing the final transcript.
8. Applying light punctuation and spacing cleanup.

## Demo Flow

1. Open `Arabic_ASR_Final_Project_Unified.ipynb`.
2. Explain the Arabic ASR problem.
3. Show the pipeline and normalization examples.
4. Show WER/CER evaluation.
5. Show results loaded from `results/`.
6. Discuss the fine-tuning experiments and Kaggle GPU run.
7. Present `Arabic_LongForm_ASR_Pipeline.pptx`.

## Optional Bonus: Clip Generation

`Clipper.py` uses transcripts to create short captioned clips from long media files. This can support educational media reuse or social-media publishing, but it is not the main ASR evaluation component.
