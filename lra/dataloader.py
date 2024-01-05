from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from transformers.data.data_collator import default_data_collator
from datasets import load_dataset, load_from_disk
import torch
import numpy as np

ROOT = "/home/mila/s/sonnery.hugo/scratch/datasets/transformers/"
VAL_PROP = 0.2
TEST_PROP = 0.1

class LRADataset(Dataset):
    def __init__(self, file_path, endless):
        self.name = "lra"
        self.split = split
        self.endless = endless
        self.seq_len = 0
        self.tasks = ["classification"]

        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)
            rd.shuffle(self.examples)

        self.curr_idx = 0


    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output["input_ids"] = torch.tensor(inst["input_ids_0"], dtype = torch.long)
        output["mask"] = (output["input_ids"] != 0).float()
        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype = torch.long)
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype = torch.long)
        return output

    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])

        if self.curr_idx >= len(self.examples):
            rd.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1

        return self.create_inst(inst)

def get_lm_dataset(dataset, tokenizer, seq_len):
    def tokenize_function(example):
        return tokenizer(example["text"])

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.rename_column("attention_mask", "mask")

    def concatenate(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tokenized_dataset.map(
        concatenate, batched=True,
    )
    return lm_dataset

class Wikitext103(Dataset):
    def __init__(self, split, endless, seq_len=1_024):
        self.name = "wikitext-103"
        self.split = split
        self.endless = endless
        self.seq_len = seq_len
        self.tasks = ["language-modeling"]
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocabulary_size = self.tokenizer.vocab_size

        try:
            split = "validation" if split == "val" else split
            self.dataset = load_from_disk(ROOT + "wikitext-103-{}/".format(split))
            self.dataset = self.dataset.with_format("torch")
            print("Loaded dataset {} from cache ({} split). Number of samples : {:,}.".format(self.__class__.__name__, self.split, len(self.dataset)))
            print("Dataset size : {:,}".format(len(self.dataset)))
        except:
            split = "validation" if split == "val" else split
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, cache_dir=ROOT + "wikitext-103/")
            lm_dataset = get_lm_dataset(dataset, self.tokenizer, self.seq_len)
            lm_dataset.save_to_disk(ROOT + "wikitext-103-{}/".format(split))
            self.dataset = lm_dataset.with_format("torch")
            print("Loaded dataset {} and saved to cache ({} split). Number of samples : {:,}.".format(self.__class__.__name__, self.split, len(self.dataset)))

        self.curr_idx = 0
        self.length = len(self.dataset)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.dataset)

    def create_inst(self, inst):
        return inst

    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.dataset[i])

        if self.curr_idx >= len(self.dataset):
            self.dataset.shuffle()
            self.curr_idx = 0
        inst = self.dataset[self.curr_idx]
        self.curr_idx += 1

        return self.create_inst(inst)

class Enwik8(Dataset):
    def __init__(self, split, endless, seq_len=512):
        self.name = "enwik8"
        self.split = split
        self.endless = endless
        self.seq_len = seq_len
        self.tasks = ["language-modeling"]

        with gzip.open(ROOT + 'enwik8.gz') as file:
            total_length = int(95e6)
            val_length = int((1 - (VAL_PROP + TEST_PROP)) * total_length)
            test_length = int((1 - TEST_PROP) * total_length)
            self.examples = np.fromstring(file.read(total_length), dtype=np.uint8)
            train, val, test = np.split(self.examples, [val_length, test_length])
            self.examples = torch.from_numpy(train if split == "train" else val if split == "val" else test)

        self.vocabulary_size = int(torch.max(self.examples) - 1)
        self.curr_idx = 0
        self.length = len(self.examples)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return math.ceil(len(self.examples) / self.seq_len)

    def create_inst(self):
        output = {}
        rand_start = torch.randint(0, self.examples.size(0) - self.seq_len - 1, (1,))
        output["input_ids"] = self.examples[rand_start: rand_start + self.seq_len].long()
        output["mask"] = True
        return output

    def __getitem__(self, i):
        return self.create_inst()

datasets = {
    "text": lambda split, endless: LRADataset(f"/long-range-arena/text.{split if split != 'val' else 'dev'}.pickle", endless),
    "listops": lambda split, endless: LRADataset(f"/long-range-arena/listops.{split if split != 'val' else 'dev'}.pickle", endless),
    "image": lambda split, endless: LRADataset(f"/long-range-arena/image.{split if split != 'val' else 'dev'}.pickle", endless),
    "retrieval": lambda split, endless: LRADataset(f"/long-range-arena/retrieval.{split if split != 'val' else 'dev'}.pickle", endless),
    "pathfinder32_baseline": lambda split, endless: LRADataset(f"/long-range-arena/pathfinder32-curv_baseline.{split if split != 'val' else 'dev'}.pickle", endless),
    "enwik8": Enwik8,
    "wikitext_103": Wikitext103,
}

# Example usage
train = Wikitext103("train", True)
val = Wikitext103("val", True)
test = Wikitext103("test", False)
custom_tasks = ["wikitext_103", "pg19", "one_billion_words", "bookcorpus"]
task = "wikitext_103"
dataset = datasets[task]
ds_iter = {
    "train": enumerate(DataLoader(dataset("train", True), batch_size = 32, drop_last = True, collate_fn=default_data_collator if task in custom_tasks else None)),
    "dev": enumerate(DataLoader(dataset("val", True), batch_size = 32, drop_last = True, collate_fn=default_data_collator if task in custom_tasks else None)),
    "test": enumerate(DataLoader(dataset("test", False), batch_size = 32, drop_last = True, collate_fn=default_data_collator if task in custom_tasks else None)),
}
