#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############## A ##############

# 2:

from nltk.corpus import wordnet
import pandas

# 3: 

study_synsets = wordnet.synsets("study")
print(study_synsets)
print(len(study_synsets))

study_synsets_noun = wordnet.synsets("study", pos=wordnet.NOUN)
print(len(study_synsets_noun))

# 4:
for study_synset in study_synsets_noun:
    print('\n')
    print(study_synset.name())
    print(study_synset.definition())
    print(study_synset.examples())

# 5:


# 6:
print('Noun lemmas:')
for study_synset in study_synsets_noun:
    print(study_synset.lemma_names())

# 7:
print(wordnet.langs())


# 8:
for study_synset in study_synsets_noun:
    print('\n')
    print(study_synset.lemma_names())
    print(study_synset.lemma_names(lang='cat'))

# 'estudiar' is the lemma of the verb 'to study' in Catalan
# we don't need to specify that the part of speech is noun 
# because the lemma is not ambiguous (unlike English, that has
# 'study' for both noun and verb).
estudiar_synsets = wordnet.synsets("estudiar",lang="cat")
for estudiar_synset in estudiar_synsets:
    print('\n')
    print(estudiar_synset.lemma_names())
    print(estudiar_synset.lemma_names(lang='cat'))

# 10:
print('i) Number of lemmas per language')
n_ss = sum(1 for _ in wordnet.all_synsets())
for lang in wordnet.langs():
    n_lemmas = sum([any(ss.lemmas(lang=lang)) for ss in wordnet.all_synsets()])
    print(lang, n_lemmas)

############# B ##############

# 11: How to approach this exercise:
    
# 11.1. Find the synsets corresponding to bass3 and bass7 in the book
# in all synsets of "bass", find the ones whose definitions match those in the book

synsets_bass = wordnet.synsets('bass')

for synset in synsets_bass:
    print(synset, synset.definition())
    print()

# 11.2 looking at the definitions, now we know these are the ones used in the book:
bass3 = synsets_bass[2]
bass7 = synsets_bass[6]

# 11.3 get hypernyms through method
hypernym_path_bass3 = bass3.hypernym_paths()[1]
hypernym_path_bass7 = bass7.hypernym_paths()[0]

for num_spaces, synset in enumerate(hypernym_path_bass3[::-1]):
    print(' ' * num_spaces, synset.lemma_names())

num_spaces = 0
for num_spaces, synset in enumerate(hypernym_path_bass7[::-1]):
    print(' ' * num_spaces, synset.lemma_names())

# 12

synsets_gift = wordnet.synsets('gift')

# note that not all senses (synsets) have a hypernym
for synset in synsets_gift:
    print(f'Synset: {synset.name()}')
    print(f'Definition: {synset.definition()}')
    for hypernym in synset.hypernyms():
        print(f'Hypernym - synset: {hypernym.name()}, lemmas: {hypernym.lemma_names()}')
    print('\n')


# 13

# 14 (Complete the code:)

def get_WordNet_relation(word1, word2):
    synsets_word1 = wordnet.synsets(word1)
    synsets_word2 = wordnet.synsets(word2)

    #relation_found = True
    for synset1 in synsets_word1:
        for synset2 in synsets_word2:
            # check hypernyms / hyponyms
            if synset2 in synset1.hypernyms():
                return print(f'{synset2} is a hypernym of {synset1}')
            # check instance hypernyms / hyponyms
            elif synset2 in synset1.instance_hyponyms():
                return print(f'{synset2} is an instance hyponym of {synset1}')
            # check Mero- / Holonyms
            elif synset1 in synset2.part_meronyms():
                return print(f'{synset1} is a part of {synset2}.')
            elif synset1 in synset2.part_holonyms():
                return print(f'{synset1} has {synset2} as a part.')
            # check antonyms:
            # check derivations:
    return print(f'I did not find a lexical relationship between {word1} and {word2} in WordNet')


examples = [('breakfast', 'meal'),
            ('meal', 'lunch'),
            ('Austen', 'author'),
            ('composer', 'Bach'),
            ('table', 'leg'),
            ('course', 'meal'),
            ('leader', 'follower'),
            ('destruction', 'destroy')]

for example in examples:
    get_WordNet_relation(example[0], example[1])
