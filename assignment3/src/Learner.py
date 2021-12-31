import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random
import json
from Classifier9001 import TweetClassifier
from DataLoader import CustomDataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def load_dicts(preprocessed_path, dataset_path):
    preprocessed_text_path = os.path.join(preprocessed_path, "tweet_texts.json")
    preprocessed_parents_path = os.path.join(preprocessed_path, "tweet_parents.json")
    preprocessed_embeddings_path = os.path.join(preprocessed_path, "tweet_embeddings.json")
    # dataset_path_path = os.path.join(dataset_path, "rumoureval-subtaskA-dev.json")
    dataset_path_path = os.path.join(dataset_path, "rumoureval-subtaskA-train.json")

    text_d = {}
    parents_d = {}
    embeddings_d = {}
    dataset_d = {}

    with open(preprocessed_text_path) as json_file:
        text_d = json.load(json_file)

    with open(preprocessed_parents_path) as json_file:
        parents_d = json.load(json_file)

    with open(preprocessed_embeddings_path) as json_file:
        embeddings_d = json.load(json_file)

    with open(dataset_path_path) as json_file:
        dataset_d = json.load(json_file)

    return text_d, parents_d, embeddings_d, dataset_d


def get_indices(task_dict, test_fraction=0.2):
    keys = list(task_dict.keys())
    values = list(task_dict.values())
    comment_l = []
    deny_l = []
    query_l = []
    support_l = []

    for index, v in enumerate(values):
        if v == 'comment':
            comment_l.append(index)
        elif v == 'deny':
            deny_l.append(index)
        elif v == 'query':
            query_l.append(index)
        elif v == 'support':
            support_l.append(index)

    # Balance datasets
    min_count = min(len(comment_l), len(deny_l), len(query_l), len(support_l))
    comment_l = random.sample(comment_l, min_count)
    deny_l = random.sample(deny_l, min_count)
    query_l = random.sample(query_l, min_count)
    support_l = random.sample(support_l, min_count)

    comment_train, comment_test = train_test_split(comment_l, test_size=test_fraction, shuffle=True)
    deny_train, deny_test = train_test_split(deny_l, test_size=test_fraction, shuffle=True)
    query_train, query_test = train_test_split(query_l, test_size=test_fraction, shuffle=True)
    support_train, support_test = train_test_split(support_l, test_size=test_fraction, shuffle=True)

    train_indices = comment_train + deny_train + query_train + support_train
    test_indices = comment_test + deny_test + query_test + support_test

    print("category\t", len(train_indices), "\t", len(test_indices))
    print("comment\t", len(comment_train), len(comment_test))
    print("deny\t", len(deny_train), len(deny_test))
    print("query\t", len(query_train), len(query_test))
    print("support\t", len(support_train), len(support_test))

    return train_indices, test_indices


def get_dataloaders(data_loader, subtask_A, test_fraction=0.05, batch_size=256):
    train_indices, test_indices = get_indices(subtask_A, test_fraction=test_fraction)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader


class Learner():

    def __init__(self, epochs=500, seed=42):
        self.set_all_seeds(seed=seed)
        self.epochs = epochs
        preprocessed_path = os.path.join("..", "res", "pre_processed")
        dataset_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev")
        self.state_dict_path = os.path.join("..", "res", "custom_model", "chk.pt")

        class_encode = {
            'comment': 0,
            'deny': 1,
            'query': 2,
            'support': 3
        }

        self.class_decode = {
            0: 'comment',
            1: 'deny',
            2: 'query',
            3: 'support'
        }

        self.text_d, self.parents_d, self.embeddings_d, subtask_A = load_dicts(preprocessed_path=preprocessed_path,
                                                                               dataset_path=dataset_path)

        self.data_loader = CustomDataLoader(embedding_dict=self.embeddings_d,
                                            parent_dict=self.parents_d,
                                            label_dict=subtask_A,
                                            one_hot_dict=class_encode)

        self.train_loader, self.test_loader = get_dataloaders(data_loader=self.data_loader, subtask_A=subtask_A)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = TweetClassifier()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # https://analyticsindiamag.com/ultimate-guide-to-pytorch-optimizers/
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def run_on_dataloader(self, train=False):
        if train:
            self.model.train()
            dataloader = self.train_loader
        else:
            self.model.eval()
            dataloader = self.test_loader

        total_loss = 0
        count = 0
        for x, y in dataloader:
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            if train:
                self.optimizer.zero_grad()

            output = self.model(x)
            # output = output.long()
            y = y.long()
            loss = self.criterion(output, y)
            total_loss += loss.item()
            count += 1

            if train:
                loss.backward()
                self.optimizer.step()

        if count > 0:
            total_loss /= count

        return total_loss

    def set_all_seeds(self, seed):
        # This is for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def learn(self):
        writer = SummaryWriter(comment='Veracity')
        TRAIN_LOSS = "train_loss"
        TEST_LOSS = "test_loss"
        lowest_test_loss = np.inf
        for e in trange(self.epochs):
            train_loss = self.run_on_dataloader(train=True)
            writer.add_scalar(TRAIN_LOSS, train_loss, e)
            test_loss = self.run_on_dataloader(train=False)
            writer.add_scalar(TEST_LOSS, test_loss, e)

            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
                torch.save(self.model.state_dict(), self.state_dict_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model.to(self.device)
        self.model.eval()

    def get_parents(self, tweet_id):
        parents = []
        x = self.parents_d[tweet_id]
        while x != '':
            parents.insert(0, x)
            x = self.parents_d[x]

        return parents

    def get_predicion(self, tweet_id):
        if tweet_id not in self.embeddings_d:
            return ""
        embedding = self.data_loader.get_embeddings_from_id(tweet_id)
        embedding = embedding.to(device=self.device)
        logits = self.model(embedding)
        output = torch.argmax(logits)
        output = output.cpu()
        output = output.item()
        output = self.class_decode[output]
        return output

    def evaluate(self, file_path):
        subtaskA = {}
        with open(file_path) as json_file:
            subtaskA = json.load(json_file)
        tweet_ids = subtaskA.keys()
        targets = subtaskA.values()
        preds = [self.get_predicion(x) for x in tweet_ids]
        correct = sum(x == y for x, y in zip(preds, targets))
        perc = correct / len(preds)
        return perc
    
    def create_db(self, file_path):
        subtaskA = {}
        with open(file_path) as json_file:
            subtaskA = json.load(json_file)
        tweet_ids = subtaskA.keys()
        df = pd.DataFrame(subtaskA.items(), columns = ['ID','Label'])
        preds = [self.get_predicion(x) for x in tweet_ids]
        df['Prediction']=preds
        return df


if __name__ == "__main__":
    l = Learner(epochs=100)
    #l.learn()
    l.load_model()

    file_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-dev.json")
    # file_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-train.json")

    c = l.evaluate(file_path=file_path)
    print("Correct Percentage = \t", c)

    x = l.create_db(file_path=file_path)

    print(pd.crosstab(x['Label'], x['Prediction'], margins = True))
    print('Precision: ', precision_score(x['Label'],x['Prediction'],average = None))
    print('Recall: ',recall_score(x['Label'],x['Prediction'],average = None))

    sn.heatmap(pd.crosstab(x['Label'], x['Prediction']), annot=True)
    plt.show()
