# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from Classifier import SentimentClassifier
from CustomDataLoader import CustomDataLoader

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

MAX_LEN = 128
BATCH_SIZE = 1
LR = 2e-5
MODEL_PATH = 'data/best_model_state.bin'
CSV_PATH = 'data/result.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'


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
    print("accuracy = ", acc.item())
    # return review_texts, predictions, prediction_probs, real_values


def main():
    print("Hello World")

    df_test = pd.read_csv(CSV_TEST)

    df_test['targets'] = df_test.sentiment.apply(sentiment_to_score)

    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    test_data_loader = create_data_loader(df_test, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)

    model = SentimentClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    get_predictions(model=model, data_loader=test_data_loader, dump_path=CSV_PATH, n_examples=len(df_test))


if __name__ == "__main__":
    main()
