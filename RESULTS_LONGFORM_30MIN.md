# 30+ Minute Long-Form Results

These results address the long-form requirement directly.

## Test Data

- Dataset: Google FLEURS Arabic Egypt validation split.
- Construction: 167 validation clips concatenated into one long WAV file.
- Duration: 1804.44 seconds = 30.07 minutes.
- Reference: official normalized FLEURS transcripts concatenated in the same order.
- Reference size: 3,035 words / 17,717 characters.
- Model: `ggml-base.bin` from whisper.cpp.

This is a controlled long-form benchmark with a valid reference transcript. It is not a natural podcast episode, but it satisfies the 30+ minute long-duration ASR pipeline test while preserving ground-truth evaluation.

## Results

| System | Chunk Size | Overlap | Prompt Context | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| `whisper_base_30min_no_context` | 60s | 2s | No | 0.5414 | **0.4926** | **0.2069** |
| `whisper_base_30min_context` | 60s | 2s | 24 words | 0.6250 | 0.5645 | 0.3259 |
| `whisper_base_30min_no_context_30s` | 30s | 2s | No | 0.5621 | 0.5153 | 0.2402 |

## Normalization On/Off Experiment

This completes the normalization comparison experiment. The same hypotheses were evaluated twice: once with raw text and once after Arabic normalization.

| System | Raw WER | Normalized WER | Absolute WER Reduction |
|---|---:|---:|---:|
| `whisper_base_30min_no_context` | 0.5414 | 0.4926 | 0.0487 |
| `whisper_base_30min_context` | 0.6250 | 0.5645 | 0.0605 |
| `whisper_base_30min_no_context_30s` | 0.5621 | 0.5153 | 0.0468 |

Arabic normalization improved measured WER for every system. This confirms that Arabic spelling variants, punctuation, diacritics, and letter-form differences can inflate raw WER even when the recognized words are close.

## Analysis

The best long-form configuration in this experiment was 60-second chunking with 2-second overlap and no prompt context.

Prompt context hurt performance. The likely reason is error propagation: when a chunk contains recognition mistakes, passing those words as a prompt can bias the next chunk toward incorrect wording.

The 30-second chunk size also performed worse than 60-second chunks. Shorter chunks increase the number of boundaries, which can introduce more reconstruction and boundary errors.

Arabic normalization reduced WER in every run. For example, the best run improved from raw WER 0.5414 to normalized WER 0.4926. This confirms that normalization is necessary for fair Arabic ASR evaluation.

## Conclusion

This experiment shows that the project now has a real 30+ minute ASR evaluation. The pipeline supports long audio, chunking, overlap, reconstruction, Arabic normalization, and WER/CER reporting.

The current runs do not show improvement from prompt context. Instead, they identify a better long-form configuration: 60-second chunks, 2-second overlap, no prompt context. Fine-tuning remains the next step for model-level ASR improvement.

## Experiment Status

| Experiment | Status |
|---|---|
| E0 Whisper baseline | Completed |
| E1 Chunking without context | Completed |
| E2 Chunking with context prompt | Completed |
| E3 Chunk-size comparison | Completed |
| E7 Normalization on/off comparison | Completed |
| E4 Small fine-tuned model | Completed on Kaggle GPU |
| E5 Larger fine-tuned model | Completed on Kaggle GPU |
| E6 Fine-tuning with augmentation | Completed on Kaggle GPU |

## Fine-Tuning Results E4-E6

These experiments were run on Kaggle GPU using FLEURS Arabic Egypt. The evaluation set used 20 held-out validation samples.

| Experiment | Train Samples | Augmentation | Eval Loss | Raw WER | Normalized WER | Normalized CER |
|---|---:|---:|---:|---:|---:|---:|
| E4 Small fine-tune | 50 | No | 2.0174 | 0.2899 | 0.2671 | 0.0675 |
| E5 Larger fine-tune | 150 | No | **1.6712** | **0.2769** | **0.2508** | **0.0618** |
| E6 Larger fine-tune + augmentation | 150 | Yes | 1.6823 | **0.2769** | **0.2508** | **0.0618** |

E5 improved over E4, showing that increasing the fine-tuning set from 50 to 150 Arabic samples improved transcription quality on the held-out validation subset.

E6 produced the same WER/CER as E5, with slightly worse validation loss. In this small run, augmentation did not provide a measurable improvement. This does not mean augmentation is useless; it means the selected augmentation settings and small training size did not improve this validation subset.

Important evaluation note: E4-E6 are evaluated on a 20-sample held-out FLEURS validation subset, while the 30+ minute long-form chunking experiments are evaluated on a concatenated 30.07-minute FLEURS benchmark. The fine-tuning results demonstrate model-level ASR improvement, and the long-form results demonstrate pipeline-level handling of 30+ minute audio.
