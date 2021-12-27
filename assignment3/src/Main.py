import os
import json

'''
from transformers import DistilBertTokenizer, DistilBertModel
import torch


def get_embeddings(tweet_string):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(tweet_string,
                       return_tensors="pt",
                       padding="max_length",
                       add_special_tokens=True)
    print("type input\t", type(inputs))
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print("type ouput\t", type(outputs))
    print("type last hs\t", type(last_hidden_states), "\t Size\t", last_hidden_states.size())
    return outputs, last_hidden_states

'''

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
        parents.insert(0, x)
        x = parents_dict[x]

    return parents


def view_stuff(tweet_id, text_d, parents_d):
    parents = get_parents(tweet_id=tweet_id, parents_dict=parents_d)
    for tweet in parents:
        print("\n", tweet, "\t---------------------------")
        print(text_d[tweet])

    print("\n", tweet_id, "\t---------------------------")
    print(text_d[tweet_id])


def main():
    print("Hello World!\nWelcome to Assignment3\nSDQC\n")
    preprocessed_path = os.path.join("..", "res", "pre_processed")
    text_d, parents_d = load_dicts(preprocessed_path=preprocessed_path)
    print(len(text_d))
    print(len(parents_d))
    tweet_id = "529687410611728384"
    view_stuff(tweet_id=tweet_id, text_d=text_d, parents_d=parents_d)
    #get_embeddings(text_d[tweet_id])


if __name__ == "__main__":
    main()
