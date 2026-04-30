# Demo Results

These are local demo results from a public Arabic FLEURS sample.

## Test Data

- Dataset: Google FLEURS Arabic Egypt validation split.
- Audio: 12 validation clips concatenated into one WAV file.
- Duration: 136.28 seconds.
- Reference: official normalized FLEURS transcript.
- Model used for the quick local run: `ggml-base.bin`.

This is a reproducible short demo test. The completed 30+ minute benchmark results are reported separately in `RESULTS_LONGFORM_30MIN.md`.

## Results

| System | Chunking | Overlap | Prompt Context | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| `whisper_base_fleurs_demo` | 60s | 2s | 24 words | 0.5660 | 0.4979 | 0.2109 |
| `whisper_base_no_context_fleurs_demo` | 60s | 2s | No | 0.5064 | 0.4426 | 0.1546 |
| `whisper_base_single_chunk_fleurs_demo` | 300s | 0s | No | 0.4723 | 0.4255 | 0.1287 |

## Analysis

The single-chunk run performed best on this short 136-second demo file. This is expected because the file is short enough that full transcription does not suffer from long-form instability, and it avoids chunk-boundary reconstruction errors.

The context-prompt chunked run performed worse than the no-context chunked run. This suggests that prompt context can propagate earlier recognition mistakes into later chunks, especially when the base model already makes Arabic word-level errors. This is an important experimental finding: context continuity should be evaluated, not assumed to always improve WER.

For the completed 30+ minute benchmark, see `RESULTS_LONGFORM_30MIN.md`. The long-form results confirm that chunk size, overlap, and context prompting should be evaluated rather than assumed to improve WER/CER.

## Output Files

- `results/whisper_base_fleurs_demo_predictions.jsonl`
- `results/whisper_base_fleurs_demo_metrics.json`
- `results/whisper_base_no_context_fleurs_demo_predictions.jsonl`
- `results/whisper_base_no_context_fleurs_demo_metrics.json`
- `results/whisper_base_single_chunk_fleurs_demo_predictions.jsonl`
- `results/whisper_base_single_chunk_fleurs_demo_metrics.json`
