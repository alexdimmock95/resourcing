# Basque Speech Recognition: Data Efficiency Experiment
 
Fine-tuning facebook/wav2vec2-large-xlsr-53 on Basque speech to build a CTC ASR model, systematically testing how performance scales with training data volume.
 
## Project Overview
 
How much labelled speech data does it actually take to produce a usable ASR model for a low-resource language? This experiment answers that question for Basque by training the same model architecture across four data conditions and measuring WER and CER at each point.
 
| Condition | Hours | Approx. samples |
|---|---|---|
| 10h | 10 | ~6,000 |
| 50h | 50 | ~30,500 |
| 100h | 100 | ~70,000 |
| 200h | 200 | ~140,000 |
 
## Approach
 
- Base model: facebook/wav2vec2-large-xlsr-53, a 300M parameter transformer pre-trained on 53 languages via contrastive self-supervised learning on raw audio
- Vocabulary/tokeniser: borrowed from stefan-it/wav2vec2-large-xlsr-53-basque, giving a Basque character-level CTC head (33 tokens: a-z plus Basque-specific characters ñ, í, and the word boundary token |)
- Dataset: HiTZ/composite_corpus_eu_v2.1, a composite Basque speech corpus from HiTZ (University of the Basque Country)
- Objective: CTC loss; evaluation via Word Error Rate (WER) and Character Error Rate (CER)
- Platform: Kaggle (T4 GPU, bf16 mixed precision)
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
                                    WER/CER evaluation on dev_cv split
```
 
## Key Implementation Details
 
`preprocess()`, resamples audio to 16kHz, extracts normalised `input_values` via Wav2Vec2Processor, and tokenises the transcript into character-level labels using the Basque vocabulary.
 
`CTCDataCollator`, handles variable-length sequences by padding `input_values` with 0.0 and label sequences with -100. CTC loss ignores -100 positions, so padding doesn't corrupt gradients.
 
`CTC length filter`, samples where `input_length // 320 <= label_length` are rejected before training. Without this, some samples had audio too short for their transcript length, which caused a silent loss and gradient norm collapse to zero partway through the 200h run, with no warning beforehand.
 
Chunked Parquet writing, larger datasets are written to disk in chunks of 500 to avoid PyArrow int32 overflow errors encountered with large in-memory tables. The Parquet file is deleted immediately after loading into a HuggingFace Dataset to recover disk space before training.
 
Training config (per run):
 
- Batch size: 2
- Epochs: 5
- Learning rate: 5e-4 with 200 warmup steps
- Mixed precision: bf16
- `ctc_loss_reduction`: "mean" (the HF default of "sum" produced unstable, sometimes negative loss on longer batches)
- Save/eval strategy: per epoch (early runs), by step count (later runs)
- Hub strategy: checkpoint (pushes after each epoch/step interval)
Disk management, Kaggle's 20GB working directory limit required:
 
- Deleting the Parquet file after loading (`os.remove`)
- Clearing the output directory before training (`shutil.rmtree`)
- Avoiding unnecessary Hub snapshot downloads before `trainer.train()`
## Results
 
| Hours | WER | CER |
|---|---|---|
| 10h | 0.463 | 0.087 |
| 50h | 0.489 | 0.091 |
| 100h | 0.346 | 0.063 |
| 200h | 0.25 | 0.045 |
 
WER declines steadily across the range, with no plateau reached by 200h. The jump from 100h to 200h nearly halves the error rate, suggesting more data would keep helping further past this point. 200h is the final data point for this project, not because the curve flattened, but because that's the limit of the free GPU quota available.

## Lessons Learned
 
- `ctc_loss_reduction` default ("sum") caused unstable loss on longer batches. Fixed by setting it to "mean".
- CTC length mismatch (audio too short for its transcript) caused a silent loss and gradient collapse partway through a run, with no warning beforehand. Fixed with a pre-training filter on `input_length // 320 <= label_length`.
- PyArrow int32 overflow, writing very large in-memory tables to a single Parquet file fails silently. Fixed by chunked writes.
- Disk pressure on Kaggle, model checkpoints plus Parquet files plus W&B logs fill the 20GB quota quickly. Fixed by deleting the Parquet file post-load and clearing old output directories.
- Hub resume, `trainer.train(resume_from_checkpoint=True)` looks in `output_dir` locally, not the Hub. Pass the repo ID string directly to resume from a Hub checkpoint.
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
 
## References
 
- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning for Speech Recognition, XLSR-53
- Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- HiTZ Basque NLP group, composite corpus
 
