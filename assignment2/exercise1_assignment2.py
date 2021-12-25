#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk.corpus import wordnet
import pandas as pd

data=pd.read_csv('interest_samples_in_context.csv',sep=";")

#word = 'interest'
word = 'bold'

synsets = wordnet.synsets(word)
synsets_noun = wordnet.synsets(word, pos=wordnet.NOUN)
print(len(synsets_noun))

n_synsets = len(synsets)
n_synsets_noun =len(synsets_noun)

print(f'Lemma English: {word}')
print(f'Number of synsets: {n_synsets}')
print(f'Number of noun synsets: {n_synsets_noun}\n')
print(f'Synsets details: ')
for ss in synsets:
    print(f'Name: {ss.name()}')
    print(f'Gloss: {ss.definition()}')
    print(f'Lemmas: {ss.lemmas()}')
    print(f'\n')

#contingency table#
print(f'******************\n')
print('contingency table: ')
print(f'******************\n\n')
print(pd.crosstab(data['Audrey'], data['Aakash'], margins=True))
