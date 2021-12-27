import os
import logging
import random
import pandas as pd
import numpy as np
from happytransformer import HappyTextClassification
from sklearn.model_selection import train_test_split


class GenericClassifier:

    def __init__(self, seed=42):
        logging.debug("Initializing Generi Classifier")
        self.max_seq_len = 256
        self.res_path = os.path.join("..", "res")
        self.intent_data_path = os.path.join(self.res_path, "simple_classification_dataset", "data", "")
        self.train_csv_file_path = os.path.join(self.res_path, "simple_classification_dataset", "tmp",
                                                "DoNotTouch_Train.csv")
        self.test_csv_file_path = os.path.join(self.res_path, "simple_classification_dataset", "tmp",
                                               "DoNotTouch_Test.csv")
        self.model_path = os.path.join(self.res_path, "simple_classification_dataset", "model", "")
        logging.debug("reading classes")
        self.classes = [os.path.splitext(filename)[0] for filename in os.listdir(self.intent_data_path)]
        self.mapping = {k: v for v, k in enumerate(self.classes)}

        logging.debug("creating model")
        self.model = HappyTextClassification(model_type="DISTILBERT",
                                             model_name="distilbert-base-uncased",
                                             num_labels=len(self.classes))
        logging.info("Intent Helper Initialized")
        np.random.seed(seed)

    def process_training_data(self, test_size=0.2, verbose=False):
        TEXT = "text"
        LABEL = "label"
        header = [TEXT, LABEL]

        train_intent_dataset = pd.DataFrame(columns=header)
        test_intent_dataset = pd.DataFrame(columns=header)
        for c in self.classes:
            txt = []
            class_label = self.mapping[c]
            file_path = self.intent_data_path + c + ".txt"
            with open(file_path, encoding="utf-8") as fp:
                for line in fp:
                    l = line.strip()
                    l = l.replace(",", " ")
                    l = " ".join(l.split())
                    if len(l) > 0:
                        txt.append(l)

            # the line below is a dirty lil hack to balance the classes.
            # Will also lead to faster training because most of the comments will be ignored
            txt = random.sample(txt, 330)

            train_l, test_l = train_test_split(txt, test_size=test_size, shuffle=True)
            # intent_l = [c] * len(train_l)
            intent_l = [class_label] * len(train_l)
            df_temp = pd.DataFrame(list(zip(train_l, intent_l)), columns=header)
            train_intent_dataset = train_intent_dataset.append(df_temp, ignore_index=True)

            # intent_l = [c] * len(test_l)
            intent_l = [class_label] * len(test_l)
            df_temp = pd.DataFrame(list(zip(test_l, intent_l)), columns=header)
            test_intent_dataset = test_intent_dataset.append(df_temp, ignore_index=True)

        train_intent_dataset = train_intent_dataset.sample(frac=1)
        test_intent_dataset = test_intent_dataset.sample(frac=1)

        train_intent_dataset = train_intent_dataset.reset_index(drop=True)
        test_intent_dataset = test_intent_dataset.reset_index(drop=True)

        # DUMP DO_NOT_TOUCH CSV FILES.
        train_intent_dataset.to_csv(self.train_csv_file_path, index=False)
        test_intent_dataset.to_csv(self.test_csv_file_path, index=False)

        if verbose:
            print(self.classes)
            print(self.train_intent_dataset)
            print(self.test_intent_dataset)

    def train_model(self, test_size=0.2, preprocess_data=True, verbose=False):
        if preprocess_data:
            logging.info("Preprocessing Data")
            self.process_training_data(test_size=test_size)
            logging.info("Preprocessing Done")
        # args = TCTrainArgs(num_train_epochs=5)
        logging.info("Training Model")
        self.model.train(self.train_csv_file_path)
        result = self.model.test(self.test_csv_file_path)
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

    def predict(self, sentences, verbose=False):
        logging.info("Predicting")
        predictions = []
        for sentence in sentences:
            sentence = sentence.replace(",", " ")
            p = self.model.classify_text(sentence)
            pred = int(p.label.split('_')[-1])
            prediction = self.classes[pred]
            predictions.append(prediction)
            logging.info(prediction, " ", p.score, "\t", sentence)
            if verbose:
                print(prediction, " ", p.score, "\t", sentence)

        return predictions


if __name__ == "__main__":
    sentences = [
        "Hello",
        "asdf asdfasd asdfasd",
        "sure, it is a good offer",
        "I want to sell a laptop",
        "I would like to sell something",
        "I have a macbook.",
        "It is a lenovo.",
        "It has an intel processor",
        "My asus laptop has an amd processor",
        "My laptop has an amd processor",
    ]

    gc = GenericClassifier()
    gc.train_model(verbose=True)
    # gc.load_model()
    # intents = ih.predict(sentences, verbose=True)
    # print(intents)
