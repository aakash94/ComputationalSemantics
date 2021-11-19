from gensim.models import Word2Vec
import nltk
import gensim
from nltk.corpus import wordnet
import random
import matplotlib.pyplot as plt

WORD2VEC_PATH = "C:\\Users\\Aakash\\AppData\\Roaming\\nltk_data\\models\\word2vec_sample\\pruned.word2vec.txt"


def get_hypo_hyper(word):
    hyponyms = []
    hypernyms = []
    syn_array = wordnet.synsets(word)
    if len(syn_array)<=0:
        return hyponyms, hypernyms
    
    woi = syn_array[0]
    for hyp in woi.hyponyms():
        for l in hyp.lemmas():
            hyponyms.append(l.name())

    for hyp in woi.hypernyms():
        for l in hyp.lemmas():
            hypernyms.append(l.name())

    return hyponyms, hypernyms


def get_scores(model, word, verbose=False):
    hypo_values = []
    hyper_values = []
    hypo, hyper = get_hypo_hyper(word)
    hypo = list(set(hypo))
    hyper = list(set(hyper))
    hypo2 = []
    hyper2 = []

    for w in hypo:
        if w in model.index_to_key and w != word:
            sim = model.similarity(w1=word, w2=w)
            hypo_values.append(sim)
            hypo2.append(w)

    for w in hyper:
        if w in model.index_to_key and w != word:
            sim = model.similarity(w1=word, w2=w)
            hyper_values.append(sim)
            hyper2.append(w)

    if len(hyper2) <= 0 or len(hypo2) <= 0:
        return -1, -1, -1

    hypo_score = sum(hypo_values) / len(hypo_values)
    hyper_score = sum(hyper_values) / len(hyper_values)
    hypo_max = max(hypo_values)
    hyper_max = max(hyper_values)
    diff = hypo_score - hyper_score

    if verbose:
        print("hypo :", hypo2)
        print("hypo val", hypo_values)
        print("hyper :", hyper2)
        print("hyper val", hyper_values)
        print("avg hypo score", hypo_score)
        print("avg hyper score", hyper_score)
        print("max hypo score", hypo_max)
        print("max hyper score", hyper_max)

    return hypo_max, hyper_max, diff


def get_datapoints(model, count=500, verbose=False):
    c = 0
    words_set = set()
    datapoints = []
    while c < count:
        random_word = random.sample(model.index_to_key, 1)
        random_word = random_word[0]
        if random_word not in words_set:
            hypo_max, hypr_max, diff = get_scores(model, random_word, verbose=verbose)
            if hypo_max > 0 and hypr_max > 0:
                c += 1
                words_set.add(random_word)
                datapoints.append((random_word, hypo_max, hypr_max, diff))
    return datapoints


def plot_datapoints(datapoints):
    word, hypo_score, hyper_score = list(zip(*datapoints))
    plt.plot(word, hypo_score, label="hyponym")
    plt.plot(word, hyper_score, label="hypernym")
    plt.xlabel('words')
    plt.ylabel('score')
    plt.legend()
    plt.show()

def plot_box(datapoints):
    word, syn_score, ant_score, diff = list(zip(*datapoints))
    plt.boxplot(diff)
    plt.axhline(linewidth=1, color='r')
    plt.show()

def scratch():
    # w = "weapons"
    # a, b = get_hypo_hyper(w)
    # print("Hypo : ", a)
    # print("Hyper : ", b)

    print("Scratch")
    print("About to load model")
    model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=False)
    print("Model Loaded")
    print("About to get Datapoints")
    datapoints = get_datapoints(model, count=500, verbose=False)
    print("Got Data points")
    print("Plotting Datapoints")
    #plot_datapoints(datapoints)
    plot_box(datapoints)

if __name__ == "__main__":
    scratch()
