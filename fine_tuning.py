from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio, get_dataset_config_names, get_dataset_split_names
import os

# Load base facebook model to fine-tune
ssl_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Load pretrained characters from basque fine-tune for token characters
processor = Wav2Vec2Processor.from_pretrained("stefan-it/wav2vec2-large-xlsr-53-basque")
print("\nBasque token vocabulary:")
print(processor.tokenizer.get_vocab())

# Load MCV Basque for train split
cv_train = load_dataset("HiTZ/composite_corpus_eu_v2.1", split="train", streaming=True)

sample = next(iter(cv_train))

print(sample)

def preprocess(sample):
    audio = sample["audio"]["array"]        # raw waveform
    sr = sample["audio"]["sampling_rate"]   # sample rate
    
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    labels = processor.tokenizer(sample["sentence"].lower()).input_ids

    return {"input_values": inputs.input_values, "labels": labels}    

preprocessed_audio = preprocess(sample)

print(preprocessed_audio.keys())

print("\nSample shape:")
print(preprocessed_audio["input_values"].shape)

print("\nSample labels:")
print(preprocessed_audio["labels"])

print("\nSample labels:")
print(processor.tokenizer.convert_ids_to_tokens(preprocessed_audio["labels"][:5]))

# Exit file
os._exit(0)