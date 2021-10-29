#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nltk.corpus import wordnet
import pandas as pd

# 1) Word: information.

word = 'studie'
translation = 'study'

synsets = wordnet.synsets(word, lang="nld")

n_synsets = len(synsets)

print(f'Lemma Dutch: {word} (English: {translation})')
print(f'Number of synsets: {n_synsets}')
print(f'\n')

print(f'Info about synsets: ')
print(f'\n')
for ss in synsets:
    print(f'Name: {ss.name()}')
    print(f'Gloss: {ss.definition()}')
    print(f'Lemmas Dutch: {ss.lemmas(lang="nld")}')
    print(f'\n')

# 2) Sense-annotated data: contingency table. 
# Note: We provide simulated data; you'll need to change the code 
# such that it works your CSV file *instead* of the simulated data.

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
# note the 'margins=True' parameter -- make sure you understand what it does
print(pd.crosstab(annotations['Gemma'], annotations['Lucas'], margins=True))
