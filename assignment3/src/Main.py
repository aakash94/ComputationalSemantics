from Learner import TweetClassifier
import os
import json
import pandas as pd
from sklearn.metrics import recall_score, precision_score
import seaborn as sn
import matplotlib.pyplot as plt


def silent_remove(filename):
    try:
        os.remove(filename)
    except:
        print("No file found to remove! No issues. Hopefully.")


def load_dicts():
    preprocessed_path = os.path.join("..", "res", "pre_processed")
    dataset_path = os.path.join("..", "res", "subtaska.json")
    preprocessed_text_path = os.path.join(preprocessed_path, "tweet_texts.json")
    preprocessed_parents_path = os.path.join(preprocessed_path, "tweet_parents.json")

    text_d = {}
    parents_d = {}
    subtaska_d = {}

    with open(preprocessed_text_path) as json_file:
        text_d = json.load(json_file)

    with open(preprocessed_parents_path) as json_file:
        parents_d = json.load(json_file)

    with open(dataset_path) as json_file:
        subtaska_d = json.load(json_file)

    return text_d, parents_d, subtaska_d


def official_evaluation(reference_file, submission_file):
    truth_values = json.load(open(reference_file, 'r'))
    submission = json.load(open(submission_file, 'r'))

    observed = 0
    correct = 0
    total = len(truth_values.keys())
    print(len(truth_values), 'entries in reference file')
    for reference_id in truth_values.keys():
        if reference_id in submission.keys():
            observed += 1
            if submission[reference_id] == truth_values[reference_id]:
                correct += 1
        else:
            print('unmatched entry:', reference_id, '-- no reference value for this document')

    score = correct / total

    print(observed, 'matched entries in submission')
    print(total, 'entries in reference file')

    print('sdqc accuracy:', score)
    return score


def get_predicion(tweet_text, comment_c, support_c, other_c):
    prediction = comment_c.predict(sentence=tweet_text)
    if prediction == 'comment':
        return prediction

    prediction = support_c.predict(sentence=tweet_text)
    if prediction == 'support':
        return prediction

    prediction = other_c.predict(sentence=tweet_text)

    return prediction


def evaluate(subtask, dump_path, text_d, comment_c, support_c, other_c):
    subtaskA = subtask
    tweet_ids = subtaskA.keys()
    targets = subtaskA.values()
    pred_d = {x: get_predicion(text_d[x], comment_c, support_c, other_c) for x in tweet_ids}

    silent_remove(dump_path)
    with open(dump_path, 'w') as outfile:
        json.dump(pred_d, outfile)

    preds = list(pred_d.values())
    correct = sum(x == y for x, y in zip(preds, targets))
    perc = correct / len(preds)

    df = pd.DataFrame(subtaskA.items(), columns=['ID', 'Label'])
    df['Prediction'] = preds
    return perc, df


def main():
    print("Hello World")
    seed = 42
    comment_classifier = TweetClassifier(classifier_name="comment", seed=seed)
    comment_classifier.train_model()
    comment_classifier.load_model()
    print("Trained Comment Classifier")

    support_classifier = TweetClassifier(classifier_name="support", seed=seed)
    support_classifier.train_model()
    support_classifier.load_model()
    print("Trained Support Classifier")

    other_classifier = TweetClassifier(classifier_name="other", seed=seed)
    other_classifier.train_model()
    other_classifier.load_model()
    print("Trained Other Classifier")

    dump_path = os.path.join("..", "res", "prediction.json")
    reference_path = os.path.join("..", "res", "subtaska.json")
    text_d, parents_d, subtaska_d = load_dicts()

    precision, x = evaluate(subtask=subtaska_d, dump_path=dump_path, text_d=text_d, comment_c=comment_classifier,
                            support_c=support_classifier, other_c=other_classifier)

    score = official_evaluation(reference_file=reference_path, submission_file=dump_path)
    print("\n\n\nFINAL SCORE :\t", score)

    print(pd.crosstab(x['Label'], x['Prediction'], margins=True))
    print('Precision: ', precision_score(x['Label'], x['Prediction'], average=None))
    print('Recall: ', recall_score(x['Label'], x['Prediction'], average=None))

    sn.heatmap(pd.crosstab(x['Label'], x['Prediction']), annot=True)
    plt.show()


if __name__ == "__main__":
    main()
