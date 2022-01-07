import os
import logging
import pandas as pd
import numpy as np
import random
import torch
from happytransformer import HappyTextClassification
from sklearn.model_selection import train_test_split


def silentremove(filename):
    try:
        os.remove(filename)
    except:
        print("No file found to remove! No issues. Hopefully.")


def create_dir(dir_path):
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path)


class TweetClassifier:

    def __init__(self, classifier_name, seed=42):
        logging.debug("Initializing intent helper")
        self.set_all_seeds(seed=seed)
        self.max_seq_len = 256
        res_path = os.path.join("..", "res", "simple_classification_dataset", "")
        self.data_path = os.path.join(res_path, "data", classifier_name, "")
        self.train_csv_file_path = os.path.join(res_path, "tmp", classifier_name, "DoNotTouch_Train.csv")
        self.test_csv_file_path = os.path.join(res_path, "tmp", classifier_name, "DoNotTouch_Test.csv")
        self.model_path = os.path.join(res_path, "custom_model", classifier_name, "")
        create_dir(self.model_path)
        logging.debug("reading classes")
        self.classes = [os.path.splitext(filename)[0] for filename in os.listdir(self.data_path)]
        self.mapping = {k: v for v, k in enumerate(self.classes)}

        logging.debug("creating model")
        self.model = HappyTextClassification(model_type="DISTILBERT",
                                             model_name="distilbert-base-uncased",
                                             num_labels=len(self.classes))

        logging.info("Initialization Complete")

    def set_all_seeds(self, seed):
        # This is for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def process_training_data(self, test_size=0.1, verbose=False):
        TEXT = "text"
        LABEL = "label"
        header = [TEXT, LABEL]

        train_dataset = pd.DataFrame(columns=header)
        test_dataset = pd.DataFrame(columns=header)
        for c in self.classes:
            txt = []
            class_label = self.mapping[c]
            file_path = self.data_path + c + ".txt"
            with open(file_path, encoding="utf-8") as fp:
                for line in fp:
                    txt.append(line)

            train_l, test_l = train_test_split(txt, test_size=test_size, shuffle=True)
            # intent_l = [c] * len(train_l)
            intent_l = [class_label] * len(train_l)
            df_temp = pd.DataFrame(list(zip(train_l, intent_l)), columns=header)
            train_dataset = train_dataset.append(df_temp, ignore_index=True)

            # intent_l = [c] * len(test_l)
            intent_l = [class_label] * len(test_l)
            df_temp = pd.DataFrame(list(zip(test_l, intent_l)), columns=header)
            test_dataset = test_dataset.append(df_temp, ignore_index=True)

        train_dataset = train_dataset.sample(frac=1)
        test_dataset = test_dataset.sample(frac=1)

        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        # DUMP DO_NOT_TOUCH CSVs
        silentremove(self.train_csv_file_path)
        silentremove(self.test_csv_file_path)
        train_dataset.to_csv(self.train_csv_file_path, index=False)
        test_dataset.to_csv(self.test_csv_file_path, index=False)

        if verbose:
            print(self.classes)
            print(train_dataset)
            print(test_dataset)

    def train_model(self, test_size=0.1, preprocess_data=True, verbose=False):
        if preprocess_data:
            logging.info("Preprocessing Data")
            self.process_training_data(test_size=test_size)
            logging.info("Preprocessing Done")
        # args = TCTrainArgs(num_train_epochs=5)
        logging.info("Training Model")
        self.model.train(self.train_csv_file_path)
        result = self.model.eval(self.test_csv_file_path)
        logging.info("Training Done. Result = ", result)
        logging.info("Saving Model")
        self.model.save(self.model_path)
        logging.info("Model Saved")
        if verbose:
            print(result)

    def load_model(self):
        logging.info("Loading Model")
        self.model = HappyTextClassification(load_path=self.model_path, num_labels=len(self.classes))
        logging.info("Model Loaded")

    def predict(self, sentence, verbose=False):
        logging.info("Predicting")
        sentence = sentence.replace(",", " ")
        p = self.model.classify_text(sentence)
        pred = int(p.label.split('_')[-1])
        prediction = self.classes[pred]
        logging.info(prediction, " ", p.score, "\t", sentence)
        if verbose:
            print(prediction, " ", p.score, "\t", sentence)

        return prediction


if __name__ == "__main__":
    seed = 42
    comment_classifier = TweetClassifier(classifier_name="comment", seed=42)
    comment_classifier.train_model()
    comment_classifier.load_model()
