import os
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
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


def pre_process(data_path, folder_name='prince-toronto', cleanup=False):
    tweet_text_dict = defaultdict(str)
    tweet_parent_dict = defaultdict(str)
    tweet_embedding_dict = {}
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    folder_path = os.path.join(data_path, folder_name)
    main_tweets = os.listdir(folder_path)
    for twt in main_tweets:
        tweet_path = os.path.join(folder_path, twt)

        twt_file_name = twt + ".json"
        original_tweet_path = os.path.join(tweet_path, 'source-tweet', twt_file_name)
        with open(original_tweet_path) as json_file:
            twt_json = json.load(json_file)
            t_str = twt_json['text']
            if cleanup:
                t_str = cleanify_tweet(t_str)
                if len(t_str) <= 0:
                    continue

            tweet_text_dict[twt] = t_str
            tweet_parent_dict[twt] = str(twt_json['in_reply_to_status_id'])

            tweet_embedding_dict[twt] = model.encode(tweet_text_dict[twt]).tolist()

            if (twt_json['in_reply_to_status_id'] == None):
                tweet_parent_dict[twt] = ""

        reply_folder = os.path.join(tweet_path, 'replies')
        reply_tweets = os.listdir(reply_folder)
        for reply_tweet in reply_tweets:
            reply_path = os.path.join(reply_folder, reply_tweet)
            with open(reply_path) as json_file:
                twt_json = json.load(json_file)
                twt_id = twt_json['id']
                t_str = twt_json['text']

                if cleanup:
                    t_str = cleanify_tweet(t_str)
                    if len(t_str) <= 0:
                        continue

                tweet_text_dict[twt_id] = t_str

                tweet_parent_dict[twt_id] = str(twt_json['in_reply_to_status_id'])

                tweet_embedding_dict[twt_id] = model.encode(tweet_text_dict[twt]).tolist()

                if (twt_json['in_reply_to_status_id'] == None):
                    tweet_parent_dict[twt_id] = ""

    return tweet_text_dict, tweet_parent_dict, tweet_embedding_dict


def main(cleanup=False):
    data_path = os.path.join("..", "res", "semeval2017-task8-dataset", "rumoureval-data")
    tweet_text_dict = {}
    tweet_parent_dict = {}
    tweet_embedding_dict = {}

    rumour_topics = os.listdir(data_path)
    for rumour_topic in rumour_topics:
        text_dict, parent_dict, embedding_dict = pre_process(data_path=data_path,
                                                             folder_name=rumour_topic,
                                                             cleanup=cleanup)

        tweet_text_dict.update(text_dict)
        tweet_parent_dict.update(parent_dict)
        tweet_embedding_dict.update(embedding_dict)

    text_dump_path = os.path.join("..", "res", "pre_processed", "tweet_texts.json")
    with open(text_dump_path, 'w') as outfile:
        json.dump(tweet_text_dict, outfile)

    parent_dump_path = os.path.join("..", "res", "pre_processed", "tweet_parents.json")
    with open(parent_dump_path, 'w') as outfile:
        json.dump(tweet_parent_dict, outfile)

    parent_dump_path = os.path.join("..", "res", "pre_processed", "tweet_embeddings.json")
    with open(parent_dump_path, 'w') as outfile:
        json.dump(tweet_embedding_dict, outfile)


if __name__ == "__main__":
    main(cleanup=True)
