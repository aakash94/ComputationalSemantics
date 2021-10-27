#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk.corpus import wordnet
import pandas as pd

word = 'studie'
translation = 'study'

synsets = wordnet.synsets(word, lang="nld")

n_synsets = len(synsets)

print(f'Lemma Dutch: {word}')
print(f'Lemma English: {translation}')
print(f'Number of synsets: {n_synsets}')
print(f'Synsets details: ')
for ss in synsets:
    print(f'Name: {ss.name()}')
    print(f'Gloss: {ss.definition()}')
    print(f'Lemmas Dutch: {ss.lemmas(lang="nld")}')
    print(f'Lemmas English: {ss.lemmas()}')
    print(f'\n')

# simulate data:
gemma = np.random.uniform(low=0.1, high=50, size=(100,))
noise = np.random.uniform(low=0.1, high=40, size=(100,))
lucas = gemma + noise

gemma /= np.max(gemma)
gemma *= (n_synsets-1)

lucas /= np.max(lucas)
lucas *= (n_synsets-1)

gemma = [f's{n+1}' for n in gemma.astype(int)]
lucas = [f's{n+1}' for n in lucas.astype(int)]

annotations = pd.DataFrame({'Gemma': gemma, 'Lucas': lucas})

print(f'******************\n')
print('contingency table: ')
print(f'******************\n\n')
print(pd.crosstab(annotations['Gemma'], annotations['Lucas'], margins=True))
