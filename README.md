# Basque Speech Recognition — Data Efficiency Experiment

Fine-tuning `facebook/wav2vec2-large-xlsr-53` on Basque speech to build a CTC ASR model, systematically testing how performance scales with training data volume.

---

## Project Overview

How much labelled speech data does it actually take to produce a usable ASR model for a low-resource language? This experiment answers that question for Basque by training the same model architecture across four data conditions and measuring WER at each point.

| Condition | Hours | Approx. samples |
|-----------|-------|-----------------|
| 10h       | 10    | ~6,000          |
| 50h       | 50    | ~30,500         |
| 100h      | 100   | ~70,000         |
| Full      | —     | all available   |

---

## Approach

- **Base model:** [`facebook/wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) — 300M parameter transformer pre-trained on 53 languages via contrastive self-supervised learning on raw audio
- **Vocabulary / tokeniser:** borrowed from [`stefan-it/wav2vec2-large-xlsr-53-basque`](https://huggingface.co/stefan-it/wav2vec2-large-xlsr-53-basque), giving a Basque character-level CTC head (33 tokens: a–z plus Basque-specific characters ñ, í, and the word boundary token `|`)
- **Dataset:** [`HiTZ/composite_corpus_eu_v2.1`](https://huggingface.co/datasets/HiTZ/composite_corpus_eu_v2.1) — composite Basque speech corpus from HiTZ (University of the Basque Country)
- **Objective:** CTC loss; evaluation via Word Error Rate (WER)
- **Platform:** Kaggle (T4 GPU, bf16 mixed precision)

---

## Pipeline

```
Raw audio (any SR) → resample to 16kHz → Wav2Vec2Processor
                                              ↓
                                    Feature extraction (input_values)
                                    + tokenisation (labels)
                                              ↓
                                    Chunked Parquet write → load → delete
                                    (disk management for Kaggle 20GB limit)
                                              ↓
                                    CTCDataCollator (dynamic padding)
                                              ↓
                                    HuggingFace Trainer (bf16, T4 GPU)
                                              ↓
                                    WER evaluation on dev_cv split
```

---

## Key Implementation Details

**`preprocess()`** — resamples audio to 16kHz, extracts normalised `input_values` via `Wav2Vec2Processor`, and tokenises the transcript into character-level `labels` using the Basque vocabulary.

**`CTCDataCollator`** — handles variable-length sequences by padding `input_values` with `0.0` and label sequences with `-100`. CTC loss ignores `-100` positions, so padding doesn't corrupt gradients.

**Chunked Parquet writing** — the 50h dataset (~30,500 samples) is written to disk in chunks of 500 to avoid PyArrow int32 overflow errors encountered with large in-memory tables. The Parquet file is deleted immediately after loading into a HuggingFace `Dataset` to recover disk space before training.

**Training config (50h run):**
- Batch size: 2
- Epochs: 5
- Learning rate: 5e-4 with 200 warmup steps
- Mixed precision: bf16
- Save/eval strategy: per epoch
- Hub strategy: checkpoint (pushes after each epoch)

**Disk management** — Kaggle's 20GB working directory limit required:
1. Deleting the Parquet file after loading (`os.remove`)
2. Clearing the output directory before training (`shutil.rmtree`)
3. Avoiding unnecessary Hub snapshot downloads before `trainer.train()`

---

## Results (so far)

| Condition | Epochs completed | Final train loss | Final WER |
|-----------|-----------------|-----------------|-----------|
| 50h (run4) | 3/5            | ~9.5            | ~1.01     |

> Run cut short by Kaggle GPU quota. Remaining 2 epochs pending quota reset.

WER of ~1.01 after 3 epochs from random initialisation on the CTC head represents genuine learning — loss dropped from ~70 at epoch 1 to ~9.5 by epoch 3, with WER declining from 1.03 to 1.01 across evaluations.

---

## Status

| Stage | Status |
|-------|--------|
| 10h training | ✅ Complete |
| 50h training | 🔄 3/5 epochs done — resuming after quota reset |
| 100h training | ⏳ Planned |
| Full dataset | ⏳ Planned |
| WER curve analysis | ⏳ Planned |

---

## Lessons Learned

- **PyArrow int32 overflow** — writing >2B elements to a single Parquet file fails silently. Fixed by chunked writes.
- **Disk pressure on Kaggle** — 1.26GB model × 2 checkpoints + ~4GB Parquet + W&B logs fills 20GB quickly. Fixed by deleting the Parquet file post-load and clearing old output directories.
- **Hub resume** — `trainer.train(resume_from_checkpoint=True)` looks in `output_dir` locally, not the Hub. Pass the repo ID string directly to resume from a Hub checkpoint.

---

## Requirements

```
transformers
datasets
evaluate
jiwer
torch
pyarrow
```

Training requires a GPU with bf16 support. Developed on Google Colab (T4) and Kaggle (T4).

---

## References

- Conneau et al. (2020). [*Unsupervised Cross-lingual Representation Learning for Speech Recognition*](https://arxiv.org/abs/2006.13979) — XLSR-53
- Baevski et al. (2020). [*wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*](https://arxiv.org/abs/2006.11477)
- HiTZ Basque NLP group — composite corpus