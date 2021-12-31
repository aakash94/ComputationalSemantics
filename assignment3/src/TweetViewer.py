import os
import json
import re

def cleanify_tweet(tweet):
    # remove @mentions
    clean_tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)

    # remove #hashtags
    clean_tweet = re.sub("#[A-Za-z0-9_]+", "", clean_tweet)

    # remove http:// urls
    clean_tweet = re.sub(r'http\S+', "", clean_tweet)

    clean_tweet = " ".join(clean_tweet.split())

    return clean_tweet



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
        cleaned_up = cleanify_tweet(text_d[tweet])
        print(cleaned_up)

    print("\n", tweet_id, "\t---------------------------")
    print(text_d[tweet_id])
    cleaned_up = cleanify_tweet(text_d[tweet_id])
    print("Cleaned up:\t",cleaned_up)


def show_me_the_tweet(preprocessed_path, tweet_id):
    text_d, parents_d = load_dicts(preprocessed_path=preprocessed_path)
    view_stuff(tweet_id=tweet_id, text_d=text_d, parents_d=parents_d)


def main():
    preprocessed_path = os.path.join("..", "res", "pre_processed")
    tweet_id = "553107921081749504"
    show_me_the_tweet(preprocessed_path=preprocessed_path, tweet_id=tweet_id)


if __name__ == "__main__":
    main()
