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


def silent_remove(filename):
    try:
        os.remove(filename)
    except:
        print("No file found to remove! No issues. Hopefully.")


def load_dicts(preprocessed_path, dataset_path):
    preprocessed_text_path = os.path.join(preprocessed_path, "tweet_texts.json")
    preprocessed_parents_path = os.path.join(preprocessed_path, "tweet_parents.json")
    preprocessed_embeddings_path = os.path.join(preprocessed_path, "tweet_embeddings.json")
    dataset_paths = [os.path.join(dataset_path, "rumoureval-subtaskA-dev.json"),
                     os.path.join(dataset_path, "rumoureval-subtaskA-train.json")]

    text_d = {}
    parents_d = {}
    embeddings_d = {}
    dataset_d = {}
    temp_d = {}

    with open(preprocessed_text_path) as json_file:
        text_d = json.load(json_file)

    with open(preprocessed_parents_path) as json_file:
        parents_d = json.load(json_file)

    with open(preprocessed_embeddings_path) as json_file:
        embeddings_d = json.load(json_file)

    for d_path in dataset_paths:
        with open(d_path) as json_file:
            temp_d = json.load(json_file)
        dataset_d.update(temp_d)

    return text_d, parents_d, embeddings_d, dataset_d


def get_indices(task_dict, test_fraction=0.2):
    keys = list(task_dict.keys())
    values = list(task_dict.values())
    comment_l = []
    deny_l = []
    query_l = []
    support_l = []
    other_l = []

    for index, v in enumerate(values):
        if v == 'comment':
            comment_l.append(index)
        elif v == 'deny':
            deny_l.append(index)
        elif v == 'query':
            query_l.append(index)
        elif v == 'support':
            support_l.append(index)

    other_l = deny_l + query_l + support_l
    len_comment = len(comment_l)
    len_deny = len(deny_l)
    len_query = len(query_l)
    len_support = len(support_l)
    len_other = len(other_l)

    # Balance datasets
    min_count = min(len(comment_l), len(deny_l), len(query_l), len(support_l))
    comment_l = random.sample(comment_l, len_other)
    deny_l = random.sample(deny_l, min_count)
    query_l = random.sample(query_l, min_count)
    support_l = random.sample(support_l, min_count)

    other_train, other_test = train_test_split(other_l, test_size=test_fraction, shuffle=True)
    comment_train, comment_test = train_test_split(comment_l, test_size=test_fraction, shuffle=True)
    deny_train, deny_test = train_test_split(deny_l, test_size=test_fraction, shuffle=True)
    query_train, query_test = train_test_split(query_l, test_size=test_fraction, shuffle=True)
    support_train, support_test = train_test_split(support_l, test_size=test_fraction, shuffle=True)

    comment_train_i = comment_train + other_train
    comment_test_i = comment_test + other_test

    other_train_i = deny_train + query_train + support_train
    other_test_i = deny_test + query_test + support_test

    return comment_train_i, comment_test_i, other_train_i, other_test_i


def get_dataloaders(comment_data_loader, other_data_loader, subtask_A, test_fraction=0.05, batch_size=256):
    comment_train_i, comment_test_i, other_train_i, other_test_i = get_indices(subtask_A, test_fraction=test_fraction)

    comment_train_sampler = SubsetRandomSampler(comment_train_i)
    other_train_sampler = SubsetRandomSampler(other_train_i)
    comment_test_sampler = SubsetRandomSampler(comment_test_i)
    other_test_sampler = SubsetRandomSampler(other_test_i)

    comment_train_loader = torch.utils.data.DataLoader(comment_data_loader, batch_size=batch_size,
                                                       sampler=comment_train_sampler)
    comment_test_loader = torch.utils.data.DataLoader(comment_data_loader, batch_size=batch_size,
                                                      sampler=comment_test_sampler)
    other_train_loader = torch.utils.data.DataLoader(other_data_loader, batch_size=batch_size,
                                                     sampler=other_train_sampler)
    other_test_loader = torch.utils.data.DataLoader(other_data_loader, batch_size=batch_size,
                                                    sampler=other_test_sampler)

    return comment_train_loader, comment_test_loader, other_train_loader, other_test_loader


def official_evaluation(reference_file, submission_file):
    truth_values = json.load(open(reference_file, 'r'))
    submission = json.load(open(submission_file, 'r'))

    observed = 0
    correct = 0
    total = len(truth_values.keys())
    print(len(truth_values), 'entries in reference file')
    for reference_id in truth_values.keys():
        if reference_id in submission.keys():
            observed += 1
            if submission[reference_id] == truth_values[reference_id]:
                correct += 1
        else:
            print('unmatched entry:', reference_id, '-- no reference value for this document')

    score = correct / total

    print(observed, 'matched entries in submission')
    print(total, 'entries in reference file')

    print('sdqc accuracy:', score)
    return score


class Learner():

    def __init__(self, epochs=500, seed=42):
        self.set_all_seeds(seed=seed)
        self.epochs = epochs
        preprocessed_path = os.path.join("..", "res", "pre_processed")
        dataset_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev")
        self.comment_state_dict_path = os.path.join("..", "res", "custom_model", "comment_chk.pt")
        self.other_state_dict_path = os.path.join("..", "res", "custom_model", "other_chk.pt")

        other_class_encode = {
            'deny': 0,
            'query': 1,
            'support': 2
        }

        self.other_class_decode = {
            0: 'deny',
            1: 'query',
            2: 'support'
        }
        comment_class_encode = {
            'comment': 0,
            'deny': 1,
            'query': 1,
            'support': 1,
            'other': 1
        }

        self.comment_class_decode = {
            0: 'comment',
            1: 'other'
        }

        self.text_d, self.parents_d, self.embeddings_d, subtask_A = load_dicts(preprocessed_path=preprocessed_path,
                                                                               dataset_path=dataset_path)

        self.comment_data_loader = CustomDataLoader(embedding_dict=self.embeddings_d,
                                                    parent_dict=self.parents_d,
                                                    label_dict=subtask_A,
                                                    one_hot_dict=comment_class_encode)

        self.other_data_loader = CustomDataLoader(embedding_dict=self.embeddings_d,
                                                  parent_dict=self.parents_d,
                                                  label_dict=subtask_A,
                                                  one_hot_dict=other_class_encode)

        self.comment_train_l, self.comment_test_l, self.other_train_l, self.other_test_l = get_dataloaders(
            comment_data_loader=self.comment_data_loader, other_data_loader=self.other_data_loader, subtask_A=subtask_A)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"

        self.comment_model = TweetClassifier(num_classes=2)
        self.other_model = TweetClassifier(num_classes=3)
        self.comment_model.to(self.device)
        self.other_model.to(self.device)

        self.comment_criterion = nn.CrossEntropyLoss()
        self.other_criterion = nn.CrossEntropyLoss()
        # https://analyticsindiamag.com/ultimate-guide-to-pytorch-optimizers/
        self.comment_optimizer = torch.optim.Adam(self.comment_model.parameters())
        self.other_optimizer = torch.optim.Adam(self.other_model.parameters())

    def run_on_dataloader(self, comments_classifier=False, train=False):
        if comments_classifier:
            current_model = self.comment_model
            current_optimizer = self.comment_optimizer
            crit = self.comment_criterion
            if train:
                current_dataloader = self.comment_train_l
            else:
                current_dataloader = self.comment_train_l
        else:
            current_model = self.other_model
            current_optimizer = self.other_optimizer
            crit = self.other_criterion
            if train:
                current_dataloader = self.other_train_l
            else:
                current_dataloader = self.other_test_l

        if train:
            current_model.train()
        else:
            current_model.eval()

        total_loss = 0
        count = 0
        for x, y in current_dataloader:
            x = x.to(device=self.device)
            y = y.to(device=self.device)

            if train:
                current_optimizer.zero_grad()

            output = current_model(x)
            # output = output.long()
            y = y.long()
            loss = crit(output, y)
            total_loss += loss.item()
            count += 1

            if train:
                loss.backward()
                current_optimizer.step()

        if count > 0:
            total_loss /= count

        if comments_classifier:
            self.comment_model = current_model
        else:
            self.other_model = current_model

        return total_loss

    def set_all_seeds(self, seed):
        # This is for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def learn(self, comment_classifier=False):
        if comment_classifier:
            TAG = "Comment"
        else:
            TAG = "Other"

        writer = SummaryWriter(comment=TAG)

        TRAIN_LOSS = "train_loss"
        TEST_LOSS = "test_loss"
        lowest_test_loss = np.inf
        progress_bar = trange(self.epochs)
        for e in progress_bar:
            train_loss = self.run_on_dataloader(comments_classifier=comment_classifier, train=True)
            writer.add_scalar(TRAIN_LOSS, train_loss, e)
            test_loss = self.run_on_dataloader(comments_classifier=comment_classifier, train=False)
            writer.add_scalar(TEST_LOSS, test_loss, e)

            if test_loss < lowest_test_loss:
                lowest_test_loss = test_loss
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
                if comment_classifier:
                    torch.save(self.comment_model.state_dict(), self.comment_state_dict_path)
                else:
                    torch.save(self.other_model.state_dict(), self.other_state_dict_path)
                # print("Saved in epoch ",e)

            description = TAG + "\tlowest test loss : " + str(lowest_test_loss) + "\tcurrent test loss: " + str(
                test_loss)
            progress_bar.set_description(description)

    def load_model(self):
        self.comment_model.load_state_dict(torch.load(self.comment_state_dict_path))
        self.comment_model.to(self.device)
        self.comment_model.eval()
        self.other_model.load_state_dict(torch.load(self.other_state_dict_path))
        self.other_model.to(self.device)
        self.other_model.eval()

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
        embedding = self.comment_data_loader.get_embeddings_from_id(tweet_id)
        embedding = embedding.to(device=self.device)
        logits = self.comment_model(embedding)
        output = torch.argmax(logits)
        output = output.cpu()
        output = output.item()
        if output == 0:
            # it is comment
            output = self.comment_class_decode[output]
        else:
            logits = self.other_model(embedding)
            output = torch.argmax(logits)
            output = output.cpu()
            output = output.item()
            output = self.other_class_decode[output]

        return output

    def evaluate(self, file_path, dump_path):
        subtaskA = {}
        with open(file_path) as json_file:
            subtaskA = json.load(json_file)
        tweet_ids = subtaskA.keys()
        targets = subtaskA.values()
        pred_d = {x: self.get_predicion(x) for x in tweet_ids}

        silent_remove(dump_path)
        with open(dump_path, 'w') as outfile:
            json.dump(pred_d, outfile)

        preds = list(pred_d.values())
        correct = sum(x == y for x, y in zip(preds, targets))
        perc = correct / len(preds)
        return perc

    def create_db(self, file_path):
        subtaskA = {}
        with open(file_path) as json_file:
            subtaskA = json.load(json_file)
        tweet_ids = subtaskA.keys()
        df = pd.DataFrame(subtaskA.items(), columns=['ID', 'Label'])
        preds = [self.get_predicion(x) for x in tweet_ids]
        df['Prediction'] = preds
        return df


def main():
    # file_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-dev.json")
    # file_path = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-train.json")
    file_path = os.path.join("..", "res", "subtaska.json")

    dump_path = os.path.join("..", "res", "predictions.json")

    best_seed = 0
    best_performance = 0

    train_epochs = 100

    #progress_bar = trange(100)
    # for seed_val in progress_bar:
    seed_val = 47
    if True:
        l = Learner(epochs=train_epochs, seed=seed_val)
        l.learn(comment_classifier=True)
        l.learn(comment_classifier=False)
        c = l.evaluate(file_path=file_path, dump_path=dump_path)
        description = "best : " + str(best_performance) + "\tcurrent : " + str(c)
        # progress_bar.set_description(description)
        print(description)
        # print(seed_val, "\t", c)
        if c > best_performance:
            best_performance = c
            best_seed = seed_val

    l.load_model()
    # l = Learner(epochs=train_epochs, seed=best_seed)
    # l.learn()
    # l.evaluate(file_path=file_path, dump_path=dump_path)
    score = official_evaluation(reference_file=file_path, submission_file=dump_path)

    print("\n\n\nFINAL SCORE :\t", score)
    print("Best Seed :\t", best_seed)

    x = l.create_db(file_path=file_path)

    print(pd.crosstab(x['Label'], x['Prediction'], margins=True))
    print('Precision: ', precision_score(x['Label'], x['Prediction'], average=None))
    print('Recall: ', recall_score(x['Label'], x['Prediction'], average=None))

    sn.heatmap(pd.crosstab(x['Label'], x['Prediction']), annot=True)
    plt.show()


if __name__ == "__main__":
    main()
