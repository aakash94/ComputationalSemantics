# -*- coding: utf-8 -*-

import os
from CustomDataLoader import CustomDataLoader
from Classifier import SentimentClassifier
import transformers
from transformers import DistilBertModel, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from tqdm import trange
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# INSTRUCTIONS: You are responsible for making sure that this script outputs 

# 1) the evaluation scores of your system on the data in CSV_TEST (minimally 
# accuracy, if possible also recall and precision).

# 2) a csv file with the contents of a dataframe built from CSV_TEST that 
# contains 3 columns: the gold labels, your system's predictions, and the texts
# of the reviews.

TRIAL = False
# ATTENTION! the only change that we are supposed to do to your code
# after submission is to change 'True' to 'False' in the following line:
EVALUATE_ON_DUMMY = True

# the real thing:
CSV_TRAIN = "data/sentiment_train.csv"
CSV_VAL = "data/sentiment_val.csv"
CSV_TEST = "data/sentiment_test.csv"  # you don't have this file; we do

if TRIAL:
    CSV_TRAIN = "data/sentiment_10.csv"
    CSV_VAL = "data/sentiment_10.csv"
    CSV_TEST = "data/sentiment_10.csv"
    print('You are using your SMALL dataset!')
elif EVALUATE_ON_DUMMY:
    CSV_TEST = "data/sentiment_dummy_test_set.csv"
    print('You are using the FULL dataset, and using dummy test data! (Ok for system development.)')
else:
    print('You are using the FULL dataset, and testing on the real test data.')

RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
MODEL_PATH = 'data/best_model_state.bin'
CSV_PATH = 'data/result.csv'
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Note : It is possible to get better performance by using better pretrained language models.
Distilbert was chosen for its light weight, and because it should be easier to run/train on most computers. 
'''
PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'

#
# def sample_tests(df):
#     tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#     sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
#
#     encoding = tokenizer.encode_plus(
#         sample_txt,
#         max_length=32,
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         return_token_type_ids=False,
#         pad_to_max_length=True,
#         return_attention_mask=True,
#         return_tensors='pt',  # Return PyTorch tensors
#     )
#
#     tokens = tokenizer.tokenize(sample_txt)
#     token_ids = tokenizer.convert_tokens_to_ids(tokens)
#     ##token_ids = tokenizer.convert_tokens_to_ids(encoding['input_ids'][0])
#     print(f' Sentence: {sample_txt}')
#     print(f'   Tokens: {tokens}')
#     print(f'Token IDs: {token_ids}')
#
#     token_lens = []
#     for txt in df.text:
#         tokens = tokenizer.encode(txt, max_length=512)
#         token_lens.append(len(tokens))
#
#     sns.distplot(token_lens)
#     plt.xlim([0, 256]);
#     plt.xlabel('Token count');
#     plt.show()
#

def sentiment_to_score(senti):
    if senti == 'neg':
        return 0
    else:
        return 1


def create_data_loader(df, tokenizer, max_len=256, batch_size=8):
    ds = CustomDataLoader(reviews=df.text.to_numpy(),
                          targets=df.targets.to_numpy(),
                          tokenizer=tokenizer,
                          max_len=max_len)

    return DataLoader(ds, batch_size=batch_size, num_workers=1)


def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                n_examples):

    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        targets = torch.unsqueeze(targets, 1)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model,
               data_loader,
               loss_fn,
               device,
               n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            targets = torch.unsqueeze(targets, 1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

def val2pred(score):
    val = 'neg'
    if score > 0.5:
        val = 'pos'
    return val


def get_predictions(model, data_loader, dump_path='data/pred.csv', n_examples=1):
    model = model.eval()
    losses = []
    correct_predictions = 0
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            targets = torch.unsqueeze(targets, 1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    df = pd.DataFrame(columns=['text'])
    df['text'] = review_texts
    df['gv'] = torch.flatten(real_values)
    df['p'] = predictions
    df['gold_values'] = df.gv.apply(val2pred)
    df['predictions'] = df.p.apply(val2pred)
    df = df.drop('gv', 1)
    df = df.drop('p', 1)
    df.to_csv(dump_path)
    print(df)
    acc = correct_predictions.double() / n_examples
    print("accuracy = ", acc)
    #return review_texts, predictions, prediction_probs, real_values




def main():
    print("Hello World")
    # read files
    # sampling all the rows, to shuffle it
    df_train = pd.read_csv(CSV_TRAIN).sample(frac=1)
    print("Train CSV has ", len(df_train), " lines")
    df_test = pd.read_csv(CSV_TEST)
    print("Test CSV has ", len(df_test), " lines")
    df_val = pd.read_csv(CSV_VAL)
    print("Val CSV has ", len(df_val), " lines")

    # turn 'pos' and 'neg' to numeric values
    df_train['targets'] = df_train.sentiment.apply(sentiment_to_score)
    df_test['targets'] = df_test.sentiment.apply(sentiment_to_score)
    df_val['targets'] = df_val.sentiment.apply(sentiment_to_score)

    # print("\n\n",df_train.info(),"\n")
    # print("\n",df_val.info(),"\n\n")
    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_data_loader = create_data_loader(df_train, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

    model = SentimentClassifier(n_classes=1, pre_trained_model_name=PRE_TRAINED_MODEL_NAME)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in trange(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = val_acc

    model2 = SentimentClassifier()
    model2.load_state_dict(torch.load(MODEL_PATH))
    get_predictions(model=model2, data_loader=test_data_loader, dump_path=CSV_PATH, n_examples=len(df_test))


if __name__ == "__main__":
    main()
