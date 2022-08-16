#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:33:16 2022

@author: freda
"""


# import library

from utils import *
import pandas as pd 
import numpy as np

# import dataset 
df_full_d1 = pd.read_csv('/Users/freda/Desktop/Final/NLP/UkraineCombinedTweetsDeduped_FEB27.csv')
df_full_d1['date'] = '02/27/2022'


df_full_d2 = pd.read_csv('/Users/freda/Desktop/Final/NLP/UkraineCombinedTweetsDeduped_MAR04.csv')
df_full_d2['date'] = '03/04/2022'

df_full_d3 = pd.read_csv('/Users/freda/Desktop/Final/NLP/UkraineCombinedTweetsDeduped_MAR24.csv')
df_full_d3['date'] = '03/24/2022'


#combine all df togehter 
df_full = pd.concat([df_full_d1,df_full_d2,df_full_d3],ignore_index=True)


# english tweets only
df_full = df_full[df_full['language']== 'en']

# extract useful columns
df = df_full.loc[:,['location','followers','totaltweets','retweetcount','text','hashtags','date']]


def clean_tweet(tweet):
    import re
    tmp = tweet.lower()
    tmp = re.sub("'", "", tmp) # to avoid removing contractions in english
    tmp = re.sub("@[A-Za-z0-9_]+","", tmp) #remove @ mentions
    tmp = re.sub("#[A-Za-z0-9_]+","", tmp) # remove hashtags
    mp = re.sub(r'http\S+', '', tmp) #remove links
    tmp = re.sub(r"www.\S+", "", tmp) # remove links
    tmp = re.sub('[()!?]', ' ', tmp) # remove punctuations
    tmp = re.sub('\[.*?\]',' ', tmp) # remove punctuations
    tmp = re.sub("[^a-z0-9]"," ", tmp) # remove non-alphanumeric characters
    return tmp     


def check_word(tweet):
    tmp_f = [word_t.lower() for word_t in tweet.split(
            ) if word_t in dictionary]
    tmp_f = ' '.join(tmp_f)
    return tmp_f 

# clean text
df['body_cleaned'] = df['text'].apply(clean_tweet)
# check if words in English dictionary
df['body_cleaned'] = df['body_cleaned'].apply(check_word)

# remove stop words
df['body_cleaned_sw'] = df['body_cleaned'].apply(my_stop_words)

# steming 

df['body_cleaned_stem'] = df['body_cleaned_sw'].apply(my_stem)