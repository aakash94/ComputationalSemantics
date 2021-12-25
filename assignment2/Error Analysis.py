# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:30:07 2021

@author: audre
"""

import pandas as pd
pd.set_option('max_columns', 30)
from sklearn.metrics import accuracy_score


bert=pd.read_csv('data/ranking_bert.csv', sep='\t')
vilbert=pd.read_csv('data/ranking_vilbert.csv', sep='\t')
print(pd.crosstab(bert['colour'], bert['bert'],margins=True))
print(pd.crosstab(vilbert['colour'], vilbert['vilbert'],margins=True))
bertacc=accuracy_score(bert['colour'], bert['bert'])
vilbertacc=accuracy_score(vilbert['colour'], vilbert['vilbert'])
print('BERT accuracy: ',bertacc)
print('VilBERT accuracy: ',vilbertacc)
bertmedian=bert['rank'].median()
vilbertmedian=vilbert['rank'].median()
print('BERT median rank: ', bertmedian)
print('VilBERT median rank: ', vilbertmedian)