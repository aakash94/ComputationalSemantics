# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:07:29 2021

@author: Audrey
"""

from gensim.models import Word2Vec
import nltk
import gensim
from nltk.corpus import wordnet
import random
import matplotlib.pyplot as plt

WORD2VEC_PATH = WORD2VEC_PATH = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')

   
def get_syn_ant(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return synonyms, antonyms


def get_scores(model, word, verbose=False):
    syn_values = []
    ant_values = []
    syn, ant = get_syn_ant(word)
    syn = list(set(syn))
    ant = list(set(ant))
    syn2 = []
    ant2 = []

    for w in syn:
        if w in model.index_to_key and w != word:
            sim = model.similarity(w1=word, w2=w)
            syn_values.append(sim)
            syn2.append(w)

    for w in ant:
        if w in model.index_to_key and w != word:
            sim = model.similarity(w1=word, w2=w)
            ant_values.append(sim)
            ant2.append(w)

    if len(ant2) <= 0 or len(syn2) <= 0:
        return -1, -1, -1

    syn_score = sum(syn_values) / len(syn_values)
    ant_score = sum(ant_values) / len(ant_values)
    syn_max = max(syn_values)
    ant_max = max(ant_values)
    syn_rand = random.sample(syn_values,1)
    ant_rand = random.sample(ant_values,1)
    syn_rand = syn_rand[0]
    ant_rand = ant_rand[0]

    if verbose:
        print("------------------------")
        print("Word :", word)
        print("SYN :", syn2)
        print("syn val", syn_values)
        print("ANT :", ant2)
        print("ant val", ant_values)
        print("avg syn score", syn_score)
        print("avg ant score", ant_score)
        print("max syn score", syn_max)
        print("max ant score", ant_max)
        print("random syn score", syn_rand)
        print("random ant score", ant_rand)

    diff = syn_score - ant_score
    # diff = syn_max - ant_max
    return syn_score, ant_score, diff


def get_datapoints(model, count=500, verbose=False):
    c = 0
    words_set = set()
    datapoints = []
    while c < count:
        random_word = random.sample(model.index_to_key, 1)
        random_word = random_word[0]
        if random_word not in words_set:
            syn_max, ant_max, diff = get_scores(model, random_word, verbose=verbose)
            if syn_max > 0 and ant_max > 0:
                c += 1
                words_set.add(random_word)
                datapoints.append((random_word, syn_max, ant_max, diff))
    return datapoints


def plot_datapoints(datapoints):
    word, syn_score, ant_score, diff = list(zip(*datapoints))
    plt.plot(word, syn_score, label="synonym")
    plt.plot(word, ant_score, label="antonym")
    plt.xlabel('words')
    plt.ylabel('score')
    plt.legend()
    plt.show()

def plot_box(datapoints):
    word, syn_score, ant_score, diff = list(zip(*datapoints))
    data=[syn_score,ant_score]
    plt.boxplot(data)
    plt.ylabel('Cosine')
    plt.xticks([1, 2], ["Average Synonym", "Average Antonym"])
    plt.axhline(linewidth=1, color='r')
    plt.show()
    print('Average Synonym Cosine: ',sum(syn_score)/len(syn_score))
    print('Average Antonym Cosine: ',sum(ant_score)/len(ant_score))


def scratch():
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