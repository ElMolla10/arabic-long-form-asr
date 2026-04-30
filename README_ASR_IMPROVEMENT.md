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
| Case study | Completed controlled 30.07-minute Arabic FLEURS long-form benchmark; the same pipeline can be applied to a lecture, khutbah, podcast, interview, or audiobook. |

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

Run controlled experiments where only one major variable changes at a time. The completed experiments are:

| Experiment | Configuration | Purpose | Status |
|---|---|---|---|
| E0 | Whisper baseline on 30.07-minute FLEURS benchmark | Establish long-form baseline | Completed |
| E1 | 60s chunks, 2s overlap, no context | Test chunking without context propagation | Completed |
| E2 | 60s chunks, 2s overlap, 24-word prompt context | Test context continuity | Completed |
| E3 | 30s chunks vs 60s chunks | Test chunk-size effect | Completed |
| E4 | Whisper fine-tune, 50 training samples | Test small-data fine-tuning | Completed |
| E5 | Whisper fine-tune, 150 training samples | Test effect of more Arabic training data | Completed |
| E6 | Whisper fine-tune, 150 samples + augmentation | Test augmentation robustness | Completed |
| E7 | Raw WER vs normalized WER | Measure Arabic normalization effect | Completed |

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

Completed long-form results:

| System | Chunk Size | Overlap | Prompt Context | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| `whisper_base_30min_no_context` | 60s | 2s | No | 0.5414 | **0.4926** | **0.2069** |
| `whisper_base_30min_context` | 60s | 2s | 24 words | 0.6250 | 0.5645 | 0.3259 |
| `whisper_base_30min_no_context_30s` | 30s | 2s | No | 0.5621 | 0.5153 | 0.2402 |

Completed fine-tuning results:

| Experiment | Train Samples | Augmentation | Eval Loss | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| E4 Small fine-tune | 50 | No | 2.0174 | 0.2899 | 0.2671 | 0.0675 |
| E5 Larger fine-tune | 150 | No | **1.6712** | **0.2769** | **0.2508** | **0.0618** |
| E6 Larger fine-tune + augmentation | 150 | Yes | 1.6823 | **0.2769** | **0.2508** | **0.0618** |

Main analysis:

- Best long-form configuration: 60-second chunks, 2-second overlap, no prompt context.
- Prompt context worsened WER/CER, likely due to error propagation.
- Arabic normalization reduced measured WER for all long-form systems.
- E5 improved over E4, showing that more Arabic fine-tuning data improved held-out validation performance.
- E6 matched E5 on WER/CER but had slightly worse validation loss, so this augmentation setup did not add measurable benefit.

## How the Clipper Fits

`clipper.py` is a bonus downstream application. It uses transcripts to create short clips and subtitles, but the project grade should be defended through the ASR experiments above.
