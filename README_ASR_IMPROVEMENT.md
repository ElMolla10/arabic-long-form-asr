# ASR Improvement Focus

This project should be presented as an Arabic ASR improvement project first. The clip-generation agent can remain as an extra component, but the main evaluation is transcription quality.

## Alignment With Project Brief

| Requirement | Where it is addressed |
|---|---|
| Domain: digital media, education, content archiving | Long-form Arabic lectures/podcasts are transcribed into searchable text and reusable captions. |
| Problem: Whisper is optimized around short segments | The system explicitly chunks audio, transcribes each chunk, and reconstructs one long transcript. |
| Objective: transcribe audio >= 30 minutes | `asr_experiments.py transcribe` accepts full podcast, lecture, khutbah, interview, or audiobook files. |
| Audio segmentation | Implemented with `--chunk-seconds`, default 30 seconds. |
| Context continuity | Implemented with overlap plus previous transcript context via `--context-words`. |
| Post-processing | Implemented through overlap de-duplication, transcript merging, whitespace cleanup, and light punctuation cleanup. |
| Literature comparison | Included below and should be discussed in the report. |
| Case study | Recommended case study: religious lecture / khutbah or university lecture recording. |

## Main Research Question

How much can Arabic transcription quality improve compared with a clear Whisper baseline by using fine-tuning, Arabic text normalization, preprocessing, augmentation, and long-form audio handling?

## Required Baseline

Use Whisper before fine-tuning as the baseline.

Recommended baseline:

```bash
python asr_experiments.py transcribe \
  --manifest data/val.jsonl \
  --model ~/Documents/whisper-models/ggml-large-v3-turbo.bin \
  --system-name whisper_baseline \
  --out results/whisper_baseline_predictions.jsonl

python asr_experiments.py evaluate \
  --pred results/whisper_baseline_predictions.jsonl \
  --out results/whisper_baseline_metrics.json
```

The baseline metrics should be reported before any improved model results.

## Structured Experiments

Run controlled experiments where only one major variable changes at a time.

| Experiment | Model / Data | Augmentation | Preprocessing | Purpose |
|---|---:|---:|---:|---|
| E0 Baseline | Whisper original | No | Basic audio conversion | Establish starting WER/CER |
| E1 Small fine-tune | 1 hour Arabic data | No | Arabic normalization | Test whether small fine-tuning helps |
| E2 Medium fine-tune | 3-5 hours Arabic data | No | Arabic normalization | Test effect of more training data |
| E3 Augmented | Same as E2 | Yes | Arabic normalization | Test robustness from noise/speed/volume augmentation |
| E4 Preprocessing variant | Best model | Same as best | normalization on/off | Measure Arabic text preprocessing effect |
| E5 Long-form audio | Best model | Same as best | chunk + reconstruct | Verify performance on long lectures/podcasts |

## Arabic Text Normalization

Before WER/CER, normalize both the reference and the hypothesis:

- Remove tashkeel/diacritics.
- Normalize alef variants: `أ إ آ ٱ` to `ا`.
- Normalize `ى` to `ي`.
- Normalize `ة` to `ه` for evaluation consistency.
- Remove tatweel `ـ`.
- Remove punctuation.
- Normalize Arabic and Western digits if needed.
- Collapse repeated whitespace.

This is implemented in `asr_experiments.py`.

## Long-Form Audio Handling

Long audio should not be evaluated as one uncontrolled block. Use chunking:

- Convert audio to 16 kHz mono WAV.
- Split into fixed chunks, for example 30 seconds.
- Use small overlap, for example 2 seconds.
- Pass the last words of the previous transcript as prompt context for the next chunk.
- Transcribe each chunk.
- Shift timestamps back to the original timeline.
- Remove duplicated text caused by overlap.
- Reconstruct the full transcript in order.
- Apply light post-processing for spacing and punctuation.

This is implemented in `asr_experiments.py` with `--chunk-seconds`, `--overlap-seconds`, and `--context-words`.

Example for a 30-60 minute lecture:

```bash
python asr_experiments.py transcribe \
  --manifest data/lecture_val.jsonl \
  --model ~/Documents/whisper-models/ggml-large-v3-turbo.bin \
  --system-name whisper_baseline_longform \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --context-words 24 \
  --out results/lecture_baseline_predictions.jsonl
```

## Literature Comparison

Whisper is trained for robust multilingual speech recognition, but its decoding pipeline commonly operates on short windows of about 30 seconds. This makes long-form transcription a pipeline problem, not only a model problem: long recordings must be segmented, decoded, merged, and evaluated consistently.

The project compares the following long-form strategies:

| Method | Description | Expected strength | Weakness |
|---|---|---|---|
| Naive full-file transcription | Send the whole file directly to the ASR system | Simple | Can lose stability on long audio and gives less control over timestamps |
| Fixed chunking without context | Split into 30-second chunks and concatenate | Handles long files | Can repeat or miss words at boundaries |
| Overlap + context prompting | Use overlapping chunks and pass previous words as prompt | Better continuity across boundaries | May still propagate earlier ASR mistakes |
| Fine-tuned model + overlap/context | Use the improved ASR model inside the same long-form pipeline | Best match to domain vocabulary | Requires training data and evaluation |

In the report, cite Whisper as the main reference and explain that the project adopts the chunking/context approach because it directly addresses the long-form limitation while keeping evaluation measurable with WER/CER.

## Evaluation

Report at minimum:

- WER before normalization.
- WER after Arabic normalization.
- CER after Arabic normalization.
- Number of evaluated files.
- Examples of errors and qualitative analysis.

Suggested results table:

| System | Train data | Augmentation | Normalization | WER | CER | Notes |
|---|---:|---:|---:|---:|---:|---|
| Whisper baseline | 0h | No | Yes |  |  | Before fine-tuning |
| Fine-tuned small | 1h | No | Yes |  |  |  |
| Fine-tuned medium | 3-5h | No | Yes |  |  |  |
| Fine-tuned + augmentation | 3-5h | Yes | Yes |  |  |  |

## How the Clipper Fits

`clipper.py` is a bonus downstream application. It uses transcripts to create short clips and subtitles, but the project grade should be defended through the ASR experiments above.
