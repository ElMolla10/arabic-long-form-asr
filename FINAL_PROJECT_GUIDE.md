# Final Project Guide

## Project

**Arabic Long-Form ASR Improvement in Modern Standard Arabic**

Track:

```text
Group 2: Long-Form Transcription in Arabic (MSA)
```

Domain:

```text
Arabic media, education, broadcasting, lectures, interviews, and archived content
```

Main objective:

```text
Improve Arabic long-form transcription quality using Whisper adaptation, preprocessing, data augmentation, and WER/CER evaluation.
```

## Final Repository Contents

| File / Folder | Purpose |
|---|---|
| `Arabic_ASR_Final_Project_Unified.ipynb` | Main final notebook. Use this for the demo and grading. |
| `Arabic_LongForm_ASR_Pipeline.pptx` | Presentation deck. |
| `README.md` | Detailed project explanation, results summary, commands, and background. |
| `FINAL_PROJECT_GUIDE.md` | Short submission guide and file map. |
| `requirements.txt` | Python dependencies. |
| `asr_experiments.py` | Reusable command-line ASR pipeline for manifest creation, transcription, normalization, and WER/CER evaluation. |
| `Clipper.py` | Optional downstream clip-generation component. |
| `data/` | Demo and 30-minute long-form JSONL manifests. |
| `results/` | Completed prediction and metric files for the ASR experiments. |

## What To Submit

Submit or share the GitHub repository:

```text
https://github.com/ElMolla10/arabic-long-form-asr
```

If only specific files are requested, submit:

```text
Arabic_ASR_Final_Project_Unified.ipynb
Arabic_LongForm_ASR_Pipeline.pptx
README.md
FINAL_PROJECT_GUIDE.md
requirements.txt
asr_experiments.py
Clipper.py
data/
results/
```

## Assignment Mapping

| Requirement | Where It Is Covered |
|---|---|
| Select one project track | Notebook title/introduction and this guide |
| Define a real-world problem | Notebook problem statement and `README.md` |
| Build a complete system pipeline | Notebook pipeline sections and `asr_experiments.py` |
| Fine-tune Whisper on Arabic data | Notebook fine-tuning cells |
| Try different dataset sizes | Notebook experiment setup and fine-tuning results |
| Apply augmentation | Notebook augmentation section |
| Apply preprocessing | Arabic normalization and audio preprocessing sections |
| Evaluate using metrics | WER/CER sections and `results/` |
| Deliver professional presentation | `Arabic_LongForm_ASR_Pipeline.pptx` |
| Deliver demo | Notebook plus saved result files |

## Recommended Demo Flow

1. Open `Arabic_ASR_Final_Project_Unified.ipynb`.
2. Explain the problem: Arabic ASR underperforms because of limited labeled data and linguistic complexity.
3. Show the system pipeline: audio preprocessing, chunking, Whisper transcription, normalization, and evaluation.
4. Show Arabic normalization examples.
5. Show WER and CER metric code.
6. Show the completed results loaded from `results/`.
7. Show the fine-tuning section and mention the Kaggle GPU run.
8. Present the final findings.
9. Open `Arabic_LongForm_ASR_Pipeline.pptx` for the formal presentation.

## Running Locally

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For the command-line ASR pipeline, system dependencies are also needed:

```text
ffmpeg
ffprobe
whisper.cpp with whisper-cli
a Whisper model file, such as ggml-base.bin
```

Example evaluation command:

```bash
python asr_experiments.py evaluate \
  --pred results/whisper_base_30min_no_context_predictions.jsonl \
  --out results/check_metrics.json \
  --csv results/check_metrics.csv
```

## Kaggle / Colab Notes

For FLEURS loading, use a compatible Hugging Face datasets version:

```python
!pip uninstall -y datasets huggingface_hub
!pip install -q --no-cache-dir "datasets==3.6.0" "huggingface_hub<1.0.0" transformers accelerate evaluate jiwer soundfile librosa audiomentations
```

Restart the runtime after installation, then load FLEURS with:

```python
from datasets import load_dataset

dataset = load_dataset("google/fleurs", "ar_eg", trust_remote_code=True)
```

The full Whisper fine-tuning cells are designed for GPU environments such as Kaggle or Colab. The repository also includes completed metrics in `results/`, so the project can still be demonstrated without rerunning expensive training.

## Final Result Summary

Best long-form pipeline result in the saved benchmark:

```text
System: whisper_base_30min_no_context
Chunk size: 60 seconds
Overlap: 2 seconds
Normalized WER: 0.4926
Normalized CER: 0.2069
```

Fine-tuning experiments showed that increasing Arabic training data improved model-level WER/CER. The tested augmentation setup did not improve over the larger non-augmented fine-tune.
