# Presentation Demo

The presentation demo should show the ASR improvement pipeline first. The clipping tool can be shown at the end as a bonus.

## Demo Goal

Show that the project can transcribe a long Arabic recording and evaluate transcription quality against a reference transcript.

## Suggested Demo Flow

### 1. Show the Problem

Explain that Whisper works on short windows, while the project target is long-form audio such as:

- a 30-60 minute podcast episode
- a university lecture
- a YouTube interview
- a khutbah or religious lecture
- an audiobook chapter

### 2. Show the Input

Use a short excerpt during the live demo, for example 2-5 minutes, but explain that the same pipeline is designed for 30+ minute files.

Example manifest:

```json
{"id": "demo_lecture", "audio": "data/audio/demo_lecture.mp4", "reference": "ضع النص العربي المصحح يدويا هنا"}
```

### 3. Run Baseline Transcription

```bash
python asr_experiments.py transcribe \
  --manifest data/demo.jsonl \
  --model ~/Documents/whisper-models/ggml-large-v3-turbo.bin \
  --system-name whisper_baseline \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --context-words 24 \
  --out results/demo_baseline_predictions.jsonl
```

### 4. Run Evaluation

```bash
python asr_experiments.py evaluate \
  --pred results/demo_baseline_predictions.jsonl \
  --out results/demo_baseline_metrics.json \
  --csv results/demo_baseline_metrics.csv
```

Show:

- raw WER
- normalized WER
- normalized CER
- one qualitative error example

### 5. Show Improved Model Comparison

Run the same manifest with the improved/fine-tuned model and compare:

```bash
python asr_experiments.py transcribe \
  --manifest data/demo.jsonl \
  --model path/to/improved-whisper-model.bin \
  --system-name whisper_improved \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --context-words 24 \
  --out results/demo_improved_predictions.jsonl
```

```bash
python asr_experiments.py evaluate \
  --pred results/demo_improved_predictions.jsonl \
  --out results/demo_improved_metrics.json \
  --csv results/demo_improved_metrics.csv
```

### 6. Present Results Table

Use a table like this:

| System | Chunking | Context | Normalization | WER | CER |
|---|---|---|---|---:|---:|
| Whisper baseline | Yes | Yes | Yes |  |  |
| Improved model | Yes | Yes | Yes |  |  |

### 7. Optional Bonus Demo

Show `clipper.py` creating captioned clips from the transcript:

```bash
python clipper.py path/to/lecture.mp4 --test
```

Explain that this is a downstream media application, while the main project result is ASR quality improvement.

