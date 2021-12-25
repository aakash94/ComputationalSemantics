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


def get_parents(tweet_id, parents_dict):
    parents = []
    x = parents_dict[tweet_id]
    while x != '':
        parents.insert(0,x)
        x = parents_dict[x]

    return parents


def main():
    print("Hello World!\nWelcome to Assignment3\nSDQC\n")
    preprocessed_path = os.path.join("..", "res", "pre_processed")
    text_d, parents_d = load_dicts(preprocessed_path=preprocessed_path)
    print(len(text_d))
    print(len(parents_d))

    parents = get_parents("580341529870348288", parents_d)
    print(parents)


if __name__ == "__main__":
    main()
