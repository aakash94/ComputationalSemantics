import os
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
import re
import errno
from nltk.stem.snowball import SnowballStemmer


def camelcase_split(list_string):
    splits = []
    for s in list_string:
        splt = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)
        recombined = " ".join(splt) if len(splt) > 0 else s
        splits.append(recombined)
    return splits


def silentremove(filename):
    try:
        os.remove(filename)
    except:
        print("No file found to remove! No issues. Hopefully.")


def cleanify_tweet(tweet, id, parent_id="", subtasks=None, verbose=False):
    if subtasks is None:
        subtasks = defaultdict(str)

    clean_tweet = tweet
    snow_stemmer = SnowballStemmer(language='english')

    # remove @mentions
    mentions = re.findall("@([a-zA-Z0-9_]{1,50})", clean_tweet)
    clean_tweet = re.sub("@[A-Za-z0-9_]+", "@", clean_tweet)

    # remove #hashtags
    hashtags = re.findall("#([a-zA-Z0-9_]{1,50})", clean_tweet)
    # clean_tweet = re.sub("#[A-Za-z0-9_]+", "", clean_tweet)

    # remove http:// urls
    # urls = re.findall(r'http\S{1,50}', "", tweet)
    clean_tweet = re.sub(r'http\S+', "http:// ", clean_tweet)

    stemlist = []
    tweetlist = clean_tweet.split()
    for w in tweetlist:
        x = snow_stemmer.stem(w)
        stemlist.append(x)
    clean_tweet = " ".join(stemlist)

    # clean_tweet = " ".join(clean_tweet.split())

    # print("Mentions =\t", mentions)
    split_mentions = camelcase_split(mentions)
    # print("Split Mentions =\t", split_mentions)

    # print("Hashtags =\t", hashtags)
    split_hashtags = camelcase_split(hashtags)
    # print("Split Hashtags =\t",split_hashtags)
    # print("urls =\t", urls)
    clean_list = split_mentions + split_hashtags
    clean_list_tweet = ". ".join(clean_list)

    if len(clean_tweet) <= 0:
        if verbose:
            category = subtasks[id]
            print("\n..............")
            print(id, "\t", category, "root: ", parent_id, "\n", tweet)
            print("clean tweet\t", clean_list_tweet)

    clean_tweet = clean_tweet + clean_list_tweet

    return clean_tweet


def pre_process(folder_path, cleanup=False, subtasks=None):
    if subtasks is None:
        subtasks = defaultdict(str)

    tweet_text_dict = {}
    tweet_parent_dict = {}

    main_tweets = os.listdir(folder_path)

    for tweet_id in main_tweets:

        tweet_file_name = tweet_id + ".json"
        tweet_path = os.path.join(folder_path, tweet_id)
        original_tweet_path = os.path.join(tweet_path, 'source-tweet', tweet_file_name)

        parent_json = {}
        with open(original_tweet_path) as json_file:
            parent_json = json.load(json_file)

        tp_original_str = parent_json['text']
        tp_parent = "" if parent_json['in_reply_to_status_id'] == None else str(parent_json['in_reply_to_status_id'])
        tp_str = cleanify_tweet(tweet=tp_original_str, id=tweet_id, subtasks=subtasks) if cleanup else tp_original_str
        tweet_text_dict[tweet_id] = tp_str
        tweet_parent_dict[tweet_id] = tp_parent

        reply_folder = os.path.join(tweet_path, 'replies')
        reply_tweets = os.listdir(reply_folder)
        for reply_tweet in reply_tweets:
            reply_path = os.path.join(reply_folder, reply_tweet)
            reply_json = {}
            with open(reply_path) as json_file:
                reply_json = json.load(json_file)

            reply_id = str(reply_json['id'])
            tr_original_str = reply_json['text']
            tr_parent = "" if reply_json['in_reply_to_status_id'] == None else str(reply_json['in_reply_to_status_id'])
            tr_str = cleanify_tweet(tweet=tr_original_str, id=reply_id, parent_id=tweet_id,
                                    subtasks=subtasks) if cleanup else tr_original_str

            if tr_parent not in tweet_parent_dict:
                print("Problem tweet ", reply_id, "\t parent", tweet_id, "\t immediate parent ", tr_parent)
                tr_parent = tweet_id

            tweet_text_dict[reply_id] = tr_str
            tweet_parent_dict[reply_id] = tr_parent

    return tweet_text_dict, tweet_parent_dict


def create_dicts(cleanup=False, subtasks=None):
    if subtasks is None:
        subtasks = defaultdict(str)
    data_path = os.path.join("..", "res", "semeval2017-task8-dataset", "rumoureval-data")
    tweet_text_dict = {}
    tweet_parent_dict = {}

    rumour_topics = os.listdir(data_path)
    for rumour_topic in rumour_topics:
        print("Scraping ", rumour_topic)
        rumour_path = os.path.join(data_path, rumour_topic)
        text_dict, parent_dict = pre_process(folder_path=rumour_path, cleanup=cleanup,
                                             subtasks=subtasks)

        tweet_text_dict.update(text_dict)
        tweet_parent_dict.update(parent_dict)

    text_dump_path = os.path.join("..", "res", "pre_processed", "tweet_texts.json")
    silentremove(text_dump_path)
    with open(text_dump_path, 'w') as outfile:
        json.dump(tweet_text_dict, outfile)

    parent_dump_path = os.path.join("..", "res", "pre_processed", "tweet_parents.json")
    silentremove(parent_dump_path)
    with open(parent_dump_path, 'w') as outfile:
        json.dump(tweet_parent_dict, outfile)

def load_dicts():
    preprocessed_path = os.path.join("..", "res", "pre_processed")
    preprocessed_text_path = os.path.join(preprocessed_path, "tweet_texts.json")
    preprocessed_parents_path = os.path.join(preprocessed_path, "tweet_parents.json")

    text_d = {}
    parents_d = {}

    with open(preprocessed_text_path) as json_file:
        text_d = json.load(json_file)

    with open(preprocessed_parents_path) as json_file:
        parents_d = json.load(json_file)

    return text_d, parents_d

def create_training_data(subtasks):
    root_path = os.path.join("res", "simple_classification_dataset", "data")

    comment_list = []
    deny_list = []
    query_list = []
    support_list = []
    anti_comment_list = []
    anti_support_list = []

    text_d, parents_d = load_dicts()
    for tweet_id in subtasks:
        category = subtasks[tweet_id]
        




def main(cleanup=True, subtasks=None):
    create_dicts(cleanup=cleanup, subtasks=subtasks)
    create_training_data(subtasks=subtasks)


if __name__ == "__main__":
    file_paths = [os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-train.json"),
                  os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-dev.json"),
                  os.path.join("..", "res", "subtaska.json")]
    subtasks = {}
    temp_subtasks = {}

    for file_path in file_paths:
        with open(file_path) as json_file:
            temp_subtasks = json.load(json_file)
        subtasks.update(temp_subtasks)

    main(cleanup=True, subtasks=subtasks)
