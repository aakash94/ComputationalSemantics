import os
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
import re
import errno


def silentremove(filename):
    try:
        os.remove(filename)
    except :
        print("No file found to remove! No issues. Hopefully.")

def cleanify_tweet(tweet, id, parent_id = "", subtasks=None):

    # remove @mentions
    if subtasks is None:
        subtasks = defaultdict(str)

    clean_tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)

    # remove #hashtags
    clean_tweet = re.sub("#[A-Za-z0-9_]+", "", clean_tweet)

    # remove http:// urls
    clean_tweet = re.sub(r'http\S+', "", clean_tweet)

    clean_tweet = " ".join(clean_tweet.split())

    if len(clean_tweet )<=0:
        print("\n..............")
        category = subtasks[id]
        print(id, "\t",category ,"root: ", parent_id, "\n",tweet)
        clean_tweet = "."


    return clean_tweet


def pre_process(folder_path, cleanup=False, subtasks=None):
    if subtasks is None:
        subtasks = defaultdict(str)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    tweet_text_dict = {}
    tweet_parent_dict = {}
    tweet_embedding_dict = {}
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
        tp_embeddings = model.encode(tp_str).tolist()
        tweet_text_dict[tweet_id] = tp_str
        tweet_parent_dict[tweet_id] = tp_parent
        tweet_embedding_dict[tweet_id] = tp_embeddings

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
            tr_str = cleanify_tweet(tweet=tr_original_str, id=reply_id, parent_id=tweet_id, subtasks=subtasks) if cleanup else tr_original_str
            tr_embeddings = model.encode(tr_str).tolist()

            tweet_text_dict[reply_id] = tr_str
            tweet_parent_dict[reply_id] = tr_parent
            tweet_embedding_dict[reply_id] = tr_embeddings

    return tweet_text_dict, tweet_parent_dict, tweet_embedding_dict

def main(cleanup = False, subtasks=None):
    if subtasks is None:
        subtasks = defaultdict(str)
    data_path = os.path.join("..", "res", "semeval2017-task8-dataset", "rumoureval-data")
    tweet_text_dict = {}
    tweet_parent_dict = {}
    tweet_embedding_dict = {}

    rumour_topics = os.listdir(data_path)
    for rumour_topic in rumour_topics:
        print("Scraping ", rumour_topic)
        rumour_path = os.path.join(data_path, rumour_topic)
        text_dict, parent_dict, embedding_dict = pre_process(folder_path=rumour_path, cleanup=cleanup, subtasks=subtasks)

        tweet_text_dict.update(text_dict)
        tweet_parent_dict.update(parent_dict)
        tweet_embedding_dict.update(embedding_dict)

    text_dump_path = os.path.join("..", "res", "pre_processed", "tweet_texts.json")
    silentremove(text_dump_path)
    with open(text_dump_path, 'w') as outfile:
        json.dump(tweet_text_dict, outfile)

    parent_dump_path = os.path.join("..", "res", "pre_processed", "tweet_parents.json")
    silentremove(parent_dump_path)
    with open(parent_dump_path, 'w') as outfile:
        json.dump(tweet_parent_dict, outfile)

    embedding_dump_path = os.path.join("..", "res", "pre_processed", "tweet_embeddings.json")
    silentremove(embedding_dump_path)
    with open(embedding_dump_path, 'w') as outfile:
        json.dump(tweet_embedding_dict, outfile)


if __name__ == "__main__":
    file_path1 = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-train.json")
    file_path2 = os.path.join("..", "res", "semeval2017-task8-dataset", "traindev", "rumoureval-subtaskA-dev.json")
    subtasks = {}
    subtask1 = {}
    subtask2 = {}
    with open(file_path1) as json_file:
        subtask1 = json.load(json_file)
    with open(file_path2) as json_file:
        subtask2 = json.load(json_file)

    subtasks.update(subtask1)
    subtasks.update(subtask2)


    main(cleanup=True, subtasks=subtasks)
    # tweet = "asdf @sf @sdfs #dfdf #s works \n@sfd \nhttps://t.co this #not stays @atb"
    # clean = cleanify_tweet(tweet)
    # print(tweet)
    # print("...............")
    # print(clean)
