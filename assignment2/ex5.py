import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import scipy.stats as ss


RES_PATH = "data/"
TSV_PATH = RES_PATH + "experiment1-dataset-color-of-concrete-objects.txt"
data_header = ["entity", "colour"]
BLACK = "black"
BLUE = "blue"
BROWN = "brown"
GREEN = "green"
GREY = "grey"
ORANGE = "orange"
PINK = "pink"
PURPLE = "purple"
RED = "red"
WHITE = "white"
YELLOW = "yellow"

COLOURS = [BLACK, BLUE, BROWN, GREEN, GREY, ORANGE, PINK, PURPLE, RED, WHITE, YELLOW]

c2i = {
    BLACK: 0,
    BLUE: 1,
    BROWN: 2,
    GREEN: 3,
    GREY: 4,
    ORANGE: 5,
    PINK: 6,
    PURPLE: 7,
    RED: 8,
    WHITE: 9,
    YELLOW: 10
}


# COLOURS = [BLACK, BLUE, BROWN, GREEN, GREY, ORANGE, PINK, RED, WHITE, YELLOW]


def get_dataset(verbose=False):
    df = pd.read_csv(TSV_PATH, sep='\t', names=data_header)
    if verbose:
        print(df)
    return df


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers, verbose=False):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]
    op = word_tokens_output.mean(dim=0)
    if verbose:
        print("Hidden States = ", op)

    return op


def get_word_vector(sent, tokenizer, model, layers, idx=0, verbose=False):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    if verbose:
        print("token ids word ", token_ids_word)

    return get_hidden_states(encoded, token_ids_word, model, layers, verbose)


def get_cosine_sim(v1, v2, verbose=False):
    sim = dot(v1, v2) / ((norm(v1) * norm(v2)))
    if verbose:
        print("Cosine Similarity : ", sim)
    return sim


def get_colour_vecs(tokenizer, model, layers, verbose=False):
    col2vec = dict()
    col2vec[BLACK] = get_word_vector(BLACK, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[BLUE] = get_word_vector(BLUE, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[BROWN] = get_word_vector(BROWN, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[GREEN] = get_word_vector(GREEN, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[GREY] = get_word_vector(GREY, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[ORANGE] = get_word_vector(ORANGE, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[PINK] = get_word_vector(PINK, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[PURPLE] = get_word_vector(PURPLE, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[RED] = get_word_vector(RED, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[WHITE] = get_word_vector(WHITE, tokenizer, model, layers, idx=0, verbose=verbose)
    col2vec[YELLOW] = get_word_vector(YELLOW, tokenizer, model, layers, idx=0, verbose=verbose)

    return col2vec


def get_simlarity_scores(word1, word2, tokenizer, model, layers, verbose=False):
    v1 = get_word_vector(word1, tokenizer, model, layers, idx=0, verbose=verbose)
    v2 = get_word_vector(word2, tokenizer, model, layers, idx=0, verbose=verbose)
    sim = get_cosine_sim(v1, v2, verbose)
    if verbose:
        print(word1, "\t", word2, "\t", sim)
    return sim


def get_tokenizer_and_model(model_string="bert-base-uncased", tokenizer_string="bert-base-uncased",
                            output_hidden_States=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_string)
    model = AutoModel.from_pretrained(model_string, output_hidden_states=output_hidden_States)
    return tokenizer, model


def get_colour_vec(word, c2v, tokenizer, model, layers, verbose):
    v = get_word_vector(word, tokenizer, model, layers, idx=0, verbose=verbose)
    vec = []
    for c in COLOURS:
        c_sim = get_cosine_sim(v, c2v[c], verbose)
        vec.append(c_sim)
    if verbose:
        print(vec)
    return vec


def get_predicted_colour(word, c2v, target_colour, tokenizer, model, layers, verbose):
    rank = 0
    vec = get_colour_vec(word, c2v, tokenizer, model, layers, verbose)
    ranks = ss.rankdata(vec)
    rank = ranks[c2i[target_colour]]
    rank = len(COLOURS)-rank +1
    pos = np.argmax(vec)
    colour = COLOURS[pos]
    return colour, rank


def ex5_old():
    layers = [-4, -3, -2, -1]
    tokenizer, model = get_tokenizer_and_model(model_string="bert-base-uncased", output_hidden_States=True)
    c2v = get_colour_vecs(tokenizer, model, layers)
    word1 = "brown"
    word2 = "horse"
    sim = get_simlarity_scores(word1, word2, tokenizer, model, layers)
    print(word1, "\t", word2, "\t", sim)
    # sent = "I like cookies ."
    # sent = "cookies"
    # idx = get_word_idx(sent, sent)
    # print(idx)
    #
    # word_embedding = get_word_vector(sent, tokenizer, model, layers)
    # print(word_embedding)
    # return word_embedding


def ex5(mode='bert', verbose=False):
    target_path = RES_PATH + "ranking_" + mode + ".csv"
    if mode == 'vilbert':
        model_string = 'data/transformers4vl-vilbert'
        layers = [-6, -5, -4, -3, -2, -1]
        # target_path = RES_PATH+"no_purple"+ mode + ".csv"
    else:
        model_string = 'bert-base-uncased'
        layers = [-4, -3, -2, -1]
        # target_path = RES_PATH + "no_purple" + mode + ".csv"

    df = get_dataset(verbose=verbose)
    list_x = df[data_header[0]].to_list()
    list_y = df[data_header[1]].to_list()
    set_y = set(list_y)
    list_y_pred = []
    list_rank = []
    tokenizer, model = get_tokenizer_and_model(model_string=model_string, output_hidden_States=True)
    c2v = get_colour_vecs(tokenizer, model, layers)
    for count, word in enumerate(list_x):
        target_colour = list_y[count]
        pred, rank = get_predicted_colour(word, c2v, target_colour, tokenizer, model, layers, verbose)
        if verbose:
            print(word, "\t", pred, "\t", rank)
        list_y_pred.append(pred)
        list_rank.append(rank)
    df[mode] = list_y_pred
    df[rank] = list_rank

    df.to_csv(target_path, index=False, sep='\t')


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("data/transformers4vl-vilbert")
    # model = AutoModel.from_pretrained("data/transformers4vl-vilbert")
    #ex5(mode='vilbert')
    ex5(mode='bert')
