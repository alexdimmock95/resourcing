from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_dataset, Audio, get_dataset_config_names, get_dataset_split_names
from dataclasses import dataclass
from evaluate import load as load_metric
import os, torch

# CPU Fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load base facebook model to fine-tune
ssl_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Load pretrained characters from basque fine-tune for token characters
processor = Wav2Vec2Processor.from_pretrained("stefan-it/wav2vec2-large-xlsr-53-basque")
print("\nBasque token vocabulary:")
print(processor.tokenizer.get_vocab())

def preprocess(sample):
    '''
    Preprocess: takes a sample, separates the audio, sampling rate, input and labels
    Args: sample
    Returns: dict with input values and labels
    '''
    audio = sample["audio"]["array"]        # raw waveform
    sr = sample["audio"]["sampling_rate"]   # sample rate
    
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    labels = processor.tokenizer(sample["sentence"].lower()).input_ids

    return {"input_values": inputs.input_values, "labels": labels}    

# Load MCV Basque for train split
cv_train_10h = load_dataset("HiTZ/composite_corpus_eu_v2.1", split="train", streaming=True).take(7000)
cv_train_10h = cv_train_10h.map(preprocess)

# Validation set
cv_dev = load_dataset("HiTZ/composite_corpus_eu_v2.1", split="dev_cv", streaming=True)
cv_dev = cv_dev.map(preprocess)

'''print(preprocessed_audio.keys())

print("\nSample shape:")
print(preprocessed_audio["input_values"].shape)

print("\nSample labels:")
print(preprocessed_audio["labels"])

print("\nSample labels:")
print(processor.tokenizer.convert_ids_to_tokens(preprocessed_audio["labels"][:5]))'''

@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features):
        # Pad input_values
        input_features = [{"input_values": f["input_values"].squeeze()} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        # Pad labels with PAD token
        label_features = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, padding=True, return_tensors="pt"
        )
        # Replace PAD token id with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels
        return batch

data_collator = CTCDataCollator(processor=processor)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-basque-10h",
    per_device_train_batch_size=2,
    eval_strategy="epoch",
    fp16=False,  # Mac doesn't support this
    learning_rate=1e-4,
    warmup_steps=500,
    save_steps=400,
    logging_steps=400,
    max_steps=50,  # 7000 / 2 batch size * 3 epochs
    )

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Trainer(
    model=ssl_model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=cv_train_10h,
    eval_dataset=cv_dev,
    processing_class=processor.feature_extractor
)

trainer.train()

# Exit file
os._exit(0)