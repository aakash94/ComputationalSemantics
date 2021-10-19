'''
Data Loader
This file is responsible for
reading strings and
converting it into appropriate pytorch tensors.
'''

from transformers import DistilBertTokenizer
import torch
from torch.utils.data import Dataset


class CustomDataLoader(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len=256):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            #padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ip_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'review_text': review,
            'input_ids': ip_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)
        }
