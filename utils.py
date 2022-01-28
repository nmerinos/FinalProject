from torch.utils.data import Dataset
import torch

"""
Guided from:
https://github.com/Adapter-Hub/adapter-transformers/blob/574d0901e40ba9ec7826a4600269d828e16fe129/src/transformers/data/datasets/glue.py#L70
"""

class FormatDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenized_values, targets):
        self.tokenized_values = tokenized_values
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.targets[i]

    def __getitem__(self, i):
        # self.tokenized_values is a list
        item = {}
        for k, v in self.tokenized_values.items():
            item[k] = torch.tensor(v[i])
        item["labels"] = torch.tensor(self.targets[i])
        return item

def tokenize(tokenizer, list_values, padding="max_length", max_length=512, truncation=True):
    return tokenizer(list_values, max_length=max_length, padding=padding, truncation=truncation)
