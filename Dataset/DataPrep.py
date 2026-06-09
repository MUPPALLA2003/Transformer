import os

os.environ["HF_HOME"] = r"E:\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\hf_cache\datasets"

from datasets import Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from Tokenizers.Tokenizer import FrenchTokenizer

def build_tokenized_english2french_dataset(path_to_data_root,path_to_save,test_prop=0.005,num_workers=8,truncate=True,max_length=512,min_length=5):

    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json",truncate=truncate,max_length=max_length)
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def generator():

        for folder in sorted(os.listdir(path_to_data_root)):

            folder_path = os.path.join(path_to_data_root,folder)

            if not os.path.isdir(folder_path):

                continue

            print(f"Processing: {folder_path}")

            english_file = None
            french_file = None

            for filename in os.listdir(folder_path):

                if filename.endswith(".en"):

                    english_file = os.path.join(folder_path,filename)

                elif filename.endswith(".fr"):

                    french_file = os.path.join(folder_path,filename)

            if (english_file is None or french_file is None):

                continue

            with open(english_file,"r",encoding="utf-8",errors="ignore",) as enf, open(french_file,"r",encoding="utf-8",errors="ignore") as frf:

                for en_line, fr_line in zip(enf,frf):

                    en_line = en_line.strip()
                    fr_line = fr_line.strip()

                    if (len(en_line) == 0 or len(fr_line) == 0):

                        continue

                    yield {"english_src": en_line,"french_tgt": fr_line}

    print("Building dataset...")
    dataset = Dataset.from_generator(generator)
    dataset = dataset.train_test_split(test_size=test_prop,seed=42)

    def tokenize_batch(examples):

        src_ids = english_tokenizer(examples["english_src"],truncation=True,max_length=max_length)["input_ids"]
        tgt_ids = french_tokenizer.encode(examples["french_tgt"])

        return {"src_ids": src_ids,"tgt_ids": tgt_ids}

    dataset = dataset.map(tokenize_batch,batched=True,batch_size=1000,num_proc=num_workers)
    dataset = dataset.remove_columns(["english_src","french_tgt"])
    dataset = dataset.filter(lambda batch: [len(x) > min_length for x in batch["tgt_ids"]],batched=True)
    dataset.save_to_disk(path_to_save)
    print(f"Saved tokenized dataset to {path_to_save}")

if __name__ == "__main__":

    DATA_ROOT = r"E:\French Translation Dataset"
    TOKENIZED_DATASET_PATH = os.path.join(DATA_ROOT,"tokenized_english2french_corpus")
    build_tokenized_english2french_dataset(path_to_data_root=DATA_ROOT,path_to_save=TOKENIZED_DATASET_PATH,test_prop=0.005,num_workers=8,truncate=True,max_length=512,min_length=5)



