# -*- coding: utf-8 -*-
"""
Assignment 2, non-graded, parts C-D (word embeddings)

NOTE: we are giving you INCOMPLETE CODE. The script won't run as is. 
You need to complete it -- remember that you can run parts of the script
with 'F9' or whatever other shortcut that your operating system uses.
"""

import nltk
import numpy as np
import pandas as pd
import gensim
from nltk.corpus import wordnet
from scipy.stats import ttest_ind
from numpy import linalg as LA

###### C ##########

# 2

# finding word2vec embeddings in our harddrive

path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')

# 3

# opening this file and reading the first couple of lines

with open(path_to_word2vec_sample) as file:
    for i, line in enumerate(file):
        if i == 1:  # change these numbers to read other lines
            print(line)
        if i == 2:
            print(line)
# --> we see that a line in the file contains a word followed by a vector (for now, it's a unique string) 

# 4

# storing the information in two lists (one for words, one for vectors)

words = []
vectors = []

with open(path_to_word2vec_sample) as file:
    for i, line in enumerate(file):  # we iterate over the lines of the file
        if i == 0:
            continue
        tokens = line.split(" ")
        word = tokens[0]
        vector_str = tokens[1:]
        vector = [float(j) for j in vector_str]

        words.append(word)
        vectors.append(vector)


#print(words[:10])
#print(vectors[0][0])  # this is the first dimension of the first vector: a float
#print(len(words), len(vectors))

# transforming the vectors into a numpy array
vectors = np.array(vectors)
print(vectors.shape)  # we have one 43981 vectors (one per word); each vector has 300 dimensions

# 5

# storing words and vectors into a dictionary

word2vec = dict(zip(words, vectors))
print(word2vec["woman"])


# 6

# function to compute the norm of a vector

# we want to compute the square root of the sum of the squares
def norm(vector):
    return LA.norm(vector)


print(norm(word2vec["woman"]))
print(np.linalg.norm(word2vec["woman"]))

'''
# 7

# function to compute cosine between 2 vectors

# we first compute the dot product between vector1 and vector 2
# we'll have it as numerator
def dot(vector1, vector2):
    return 


def cosine(vector1, vector2):
    # our denominator will be the multiplication of the norms
    return 


# check with simple 2D vectors:
print(cosine([1, 0], [0, 1]))
print(cosine([0, 1], [0, 1]))
print(cosine([-1, 0], [0, 1]))

# check with words:
print(cosine(word2vec["woman"], word2vec["girl"]))
print(cosine(word2vec["water"], word2vec["fire"]))


# 8

# function that returns 10 most similar vectors in word2vec
def most_similar(vector):
    similarities = []
    # we compute the cosine similarity between our vector and all the vectors in our word2vec dictionary
    for k, v in word2vec.items():
        # we compute the cosine similarity between our vector and all the vectors in our word2vec dictionary
        sim = (k, cosine(vector, v))
        # we add the new similarity to the 'similarities' list
        similarities.append(sim)
    # we sort the obtained list of similarities by using the "sort" function
    # NOTE: you do this part also with "less sophisticated" (and more verbose) Python -- try it out
    # with "key = ..." we specify that we want to sort based on the second element 
    # with "reverse = True", we specify that we want a descending order 
    similarities.sort(key=lambda x: x[1], reverse=True)
    # we extract the first ten words in the sorted list of similarities:
    first_ten = ...  # complete
    return first_ten

# 9

# testing the function with "student" and other words

print(most_similar(word2vec["student"]))
print(most_similar(word2vec["woman"]))
print(most_similar(word2vec["water"]))

# 10

# using gensim

# loading embeddings. With gensim, we can load our embeddings directly into a dictionary
path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
word2vec_gensim = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec_sample)

print(len(word2vec_gensim.vocab))  # check the number of words we have
print(len(word2vec_gensim["woman"]))  # check the dimensions of a word vector


# computing the cosine similarity between 2 word vectors 
# we can now use the built-in function, that does all the computation for us:

# we can easily find the top N most similar vectors to a target vector:




###### D ##########

# 2
toefl = pd.read_csv('data/toeflData.tsv', sep='\t')

# get a list for all words in word2vec
w2v_vocab = list(word2vec_gensim.vocab.keys())

n_questions = 80
OOV_target = 0
OOV_answer = 0
results = []

for n_q in range(1, n_questions):

    # get row from toefl that contains the 'question';
    # extract the solution and the 'target word'

    # check whether target-word is in word2vec. If not, continue with
    # next 'question' (n_q)

    # calculate the similarity between the target and all possible synonyms
    similarity = {}
    for answer_id in []: # complete

        # get possible answer from dataframe

        # check if answer is in word2vec, if not continue with next answer
        if answer_word not in w2v_vocab:
            OOV_answer += 1
            continue

        # if answer in word2vec, check how similar it is to the target-word
        else:
            

    # get the answer_id of the most similar answer


    # add 1 to results, if it is the correct id, 0 if it is not

print(f'Performance of Word2Vec on TOEFL: {np.mean(results)}') # the part 'np.mean(results)' computes accuracy. Check that you understand why, and how it differs from the function we saw in Part 1 of the course.
# below, print the performance of a random baseline:

print(f'Out of vocabulary targets (skipped): {OOV_target}')
print(f'Out of vocabulary answers (skipped): {OOV_answer}')


# 3

# get the data

# visualize it (using boxplots or violin plots)

# Calculate the means and significance of their difference
print(f'Mean synonym-similarity: {np.mean(syn_similarities)}')
print(f'Mean hypernym-similarity: {np.mean(hyp_similarities)}')
print(f'Significant difference? t-test: p: {ttest_ind(syn_similarities, hyp_similarities)[1]}')
print('QED')

'''