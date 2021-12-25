import os
import json


def load_dicts(preprocessed_path):
    preprocessed_text_path = os.path.join(preprocessed_path, "tweet_texts.json")
    preprocessed_parents_path = os.path.join(preprocessed_path, "tweet_parents.json")
    text_d = {}
    parents_d = {}

    with open(preprocessed_text_path) as json_file:
        text_d = json.load(json_file)

    with open(preprocessed_parents_path) as json_file:
        parents_d = json.load(json_file)

    return text_d, parents_d


def main():
    print("Hello World!\nWelcome to Assignment3\nSDQC\n")
    preprocessed_path =  os.path.join("..","res", "pre_processed")
    text_d , parents_d = load_dicts(preprocessed_path=preprocessed_path)
    print(len(text_d))
    print(len(parents_d))


if __name__ == "__main__":
    main()
