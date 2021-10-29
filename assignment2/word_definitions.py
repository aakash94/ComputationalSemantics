# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:17:05 2021

@author: Audrey
"""
from nltk.corpus import wordnet
import pandas

word_synsets_noun = wordnet.synsets("interest", pos=wordnet.NOUN)
print(len(word_synsets_noun))

count=1
for word_synset in word_synsets_noun:
    print('\n')
    print('Interest Definition ', count,':')
    print(word_synset.name())
    print(word_synset.definition())
    print(word_synset.examples())
    count=count+1