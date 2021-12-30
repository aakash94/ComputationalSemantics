import numpy as np
from torch.utils.data import Dataset
import torch


class CustomDataLoader(Dataset):

    def __init__(self, embedding_dict, parent_dict, label_dict, one_hot_dict):
        self.embeddings = embedding_dict
        self.parents = parent_dict
        self.labels = label_dict
        self.one_hot_classes = one_hot_dict
        self.tweet_ids = list(self.labels)

    def get_parents(self, tweet_id):
        parents = []
        x = self.parents[tweet_id]
        while x != '':
            parents.insert(0, x)
            x = self.parents[x]

        return parents

    def __getitem__(self, index):
        tweet_id = self.tweet_ids[index]
        tweet_embedding = self.embeddings[tweet_id]
        label = self.labels[tweet_id]
        target = self.one_hot_classes[label]
        # target = np.array(target)
        # target = torch.FloatTensor(target)

        parent_embedding = tweet_embedding
        parent_tweet_ids = self.get_parents(tweet_id)

        if len(parent_tweet_ids) > 0:
            # parent tweet exists
            parent_tweet_id = parent_tweet_ids[0]
            parent_embedding = self.embeddings[parent_tweet_id]

        combined_embedding = parent_embedding + tweet_embedding
        #combined_embedding = np.array(combined_embedding)
        combined_embedding = torch.FloatTensor(combined_embedding)

        # Convert into torch Tensor
        return combined_embedding, target

    def __len__(self):
        return len(self.labels)
