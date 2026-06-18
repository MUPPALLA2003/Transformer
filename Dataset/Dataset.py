import os
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np
 
os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\hf_cache\datasets"
 
from Tokenizers.Tokenizer import FrenchTokenizer
 
 
class TranslationCollator:

    def __init__(self,src_tokenizer,tgt_tokenizer):

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __call__(self, batch):

        src_ids = [torch.tensor(i["src_ids"], dtype=torch.long) for i in batch]
        tgt_ids = [torch.tensor(i["tgt_ids"], dtype=torch.long) for i in batch]
        src_pad_token = self.src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
        src_pad_mask = (src_padded == src_pad_token)
        tgt_pad_token = self.tgt_tokenizer.special_tokens_dict["[PAD]"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)
        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()
        tgt_pad_mask = (input_tgt == tgt_pad_token)
        output_tgt[output_tgt == tgt_pad_token] = -100

        return {"src_input_ids": src_padded,"src_pad_mask": src_pad_mask,"tgt_input_ids": input_tgt,"tgt_pad_mask": tgt_pad_mask,"tgt_outputs": output_tgt}
 
def add_length_column(dataset, num_proc=8):

    print("Computing src_len column...")
    dataset = dataset.map(lambda batch: {"src_len": [len(x) for x in batch["src_ids"]]},batched=True,batch_size=1000,num_proc=num_proc)
    
    return dataset
 
class LengthBucketSampler(Sampler):
   
    def __init__(self, dataset, batch_size, bucket_size_multiplier=100, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
 
        if "src_len" in dataset.column_names:

            self.lengths = dataset["src_len"]

        else:

            self.lengths = [len(x) for x in dataset["src_ids"]]
 
    def set_epoch(self, epoch):

        self.epoch = epoch
 
    def __iter__(self):

        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.arange(len(self.dataset))
 
        if self.shuffle:

            rng.shuffle(indices)
 
        for bucket_start in range(0, len(indices), self.bucket_size):

            bucket_indices = indices[bucket_start: bucket_start + self.bucket_size]
            bucket_indices = sorted(bucket_indices, key=lambda i: self.lengths[int(i)])

            yield from bucket_indices
 
    def __len__(self):

        return len(self.dataset)
 
 
def build_dataloader(dataset, src_tokenizer, tgt_tokenizer, batch_size,num_workers=8, bucket=True, shuffle=True, seed=0):

    collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
 
    if bucket:

        sampler = LengthBucketSampler(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)

        return DataLoader(dataset,batch_size=batch_size,sampler=sampler,collate_fn=collate_fn,num_workers=num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=num_workers > 0)
 
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn,num_workers=num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=num_workers > 0)
  
if __name__ == "__main__":
 
    DATA_ROOT = r"E:\French Translation Dataset"
    TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT, "tokenized_english2french_corpus")
    LENGTH_AUGMENTED_PATH = os.path.join(DATA_ROOT, "tokenized_english2french_corpus_with_lengths")
    tgt_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json", truncate=True, max_length=512)
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    print("Loading tokenized dataset...")
    dataset = load_from_disk(TOKENIZED_DATASET_PATH)
    dataset = add_length_column(dataset, num_proc=8)
    dataset.save_to_disk(LENGTH_AUGMENTED_PATH)
    print(f"Saved length-augmented dataset to {LENGTH_AUGMENTED_PATH}")
    train_loader = build_dataloader(dataset["train"], src_tokenizer, tgt_tokenizer,batch_size=32, num_workers=8, bucket=True, shuffle=True)
    val_loader = build_dataloader(dataset["test"], src_tokenizer, tgt_tokenizer,batch_size=32, num_workers=4, bucket=True, shuffle=False)
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})





    