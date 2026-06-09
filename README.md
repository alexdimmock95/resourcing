# Basque Speech Recognition — Data Efficiency Experiment

Fine-tuning `facebook/wav2vec2-large-xlsr-53` on Basque speech data to build a CTC ASR model, with a systematic data efficiency curve experiment across four training conditions.

---

## Project Overview

This project investigates how much labelled Basque speech data is needed to produce a usable ASR model using transfer learning from a massively multilingual SSL baseline. The core question: **does performance scale predictably with data volume?**

| Condition | Hours | Approx. samples |
|-----------|-------|-----------------|
| 10h       | 10    | ~7,000          |
| 50h       | 50    | ~35,000         |
| 100h      | 100   | ~70,000         |
| Full      | —     | all available   |

---

## Approach

- **Base model:** [`facebook/wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) — a 300M parameter transformer pre-trained on 53 languages via contrastive self-supervised learning on raw audio
- **Vocabulary / tokeniser:** borrowed from [`stefan-it/wav2vec2-large-xlsr-53-basque`](https://huggingface.co/stefan-it/wav2vec2-large-xlsr-53-basque), giving a Basque character-level CTC head
- **Dataset:** [`HiTZ/composite_corpus_eu_v2.1`](https://huggingface.co/datasets/HiTZ/composite_corpus_eu_v2.1) — a composite Basque speech corpus from HiTZ (University of the Basque Country)
- **Objective:** CTC loss; evaluation via Word Error Rate (WER)

---

## Pipeline

```
Raw audio (any SR) → resample to 16kHz → Wav2Vec2Processor
                                              ↓
                                    Feature extraction (input_values)
                                    + tokenisation (labels)
                                              ↓
                                    CTCDataCollator (dynamic padding)
                                              ↓
                                    HuggingFace Trainer (fp16, T4 GPU)
                                              ↓
                                    WER evaluation on dev_cv split
```

---

## Key Implementation Details

**`preprocess()`** — maps each dataset sample to `input_values` (normalised waveform) and `labels` (character token IDs from the Basque vocabulary).

**`CTCDataCollator`** — handles variable-length sequences by padding `input_values` with `0.0` and label sequences with `-100` (which CTC loss ignores, so padding doesn't corrupt gradients).

**Training config (10h run):**
- Batch size: 2 (gradient accumulation used to simulate larger batches on T4)
- Max steps: 10,500 (~3 epochs over 7,000 samples at batch size 2)
- Learning rate: 1e-4 with 500 warmup steps
- Mixed precision: fp16

---

## Status

| Stage | Status |
|-------|--------|
| Smoke test (50 steps, CPU) | ✅ Complete — pipeline runs end to end |
| 10h training (T4 GPU, Colab) | 🔄 In progress |
| 50h / 100h / full conditions | ⏳ Planned |
| WER curve analysis | ⏳ Planned |

> **Smoke test result:** Loss 439, WER 100% after 50 steps — expected at this scale, confirms the pipeline is correctly wired.

---

## Requirements

```
transformers
datasets
evaluate
jiwer
torch
```

```bash
pip install transformers datasets evaluate jiwer
```

Training requires a GPU. Developed and tested on Google Colab (T4).

---

## Motivation

Basque (*Euskara*) is a language isolate with no known relatives — and a low-resource language in ASR terms despite active community use. Robust open-source ASR for Basque has practical value for transcription tooling, accessibility, and language preservation. This experiment contributes a reproducible data efficiency benchmark using publicly available data and models.

---

## References

- Conneau et al. (2020). [*Unsupervised Cross-lingual Representation Learning for Speech Recognition*](https://arxiv.org/abs/2006.13979) — XLSR-53
- Baevski et al. (2020). [*wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*](https://arxiv.org/abs/2006.11477)
- HiTZ Basque NLP group — composite corpus