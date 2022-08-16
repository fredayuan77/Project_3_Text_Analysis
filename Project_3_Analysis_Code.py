#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:01:09 2022

@author: liuyaxin
"""

#%% nltk 
import pandas as pd
import numpy as np
import statistics

import warnings
warnings.filterwarnings("ignore")
#%%%
## nltk engine
#import data: df1
#path = "/Users/liuyaxin/Desktop/LYX/Columbia/Natural Language Processing/project/"
#df_1 = pd.read_csv(path + "FEB27_cleaned.csv")

def fetch_senti(var):
    from nltk.sentiment import SentimentIntensityAnalyzer
    senti_engine = SentimentIntensityAnalyzer()
    tmp = senti_engine.polarity_scores(var)["compound"]
    return tmp

# check sentiment score

##df_1["nltk_sentiment"] = df_1['clean'].apply(fetch_senti)

# TEXTBLOB: another sentiment engine

    # polarity
def fetch_polarity_blob(var): # The polarity score is a float within the range [-1.0, 1.0]. 
    from textblob import TextBlob
    tmp = TextBlob(str(var)).sentiment.polarity  # convert everything to str
    return tmp

    # subjectivity
def fetch_subjective_blob(var): # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    from textblob import TextBlob
    tmp = TextBlob(str(var)).sentiment.subjectivity  # convert everything to str
    return tmp

#df_1["textblob_polarity"] = df_1['clean'].apply(fetch_polarity_blob)

#df_1["textblob_subjectivity"] = df_1['clean'].apply(fetch_subjective_blob)

#df_1.to_csv(path + 'df1_sentiment_analyzed.csv',header=True, index=True)

#%% df2
#df_2 = pd.read_csv(path + "MARCH04_cleaned.csv")

# check sentiment score
def fetch_senti(var):
    from nltk.sentiment import SentimentIntensityAnalyzer
    senti_engine = SentimentIntensityAnalyzer()
    tmp = senti_engine.polarity_scores(str(var))["compound"]
    return tmp

#df['score'] = df['review'].apply(lambda review: senti_engine.polarity_scores(str(review)))


#df_2["nltk_sentiment"] = df_2['clean'].apply(fetch_senti)

#df_2["textblob_polarity"] = df_2['clean'].apply(fetch_polarity_blob)

#df_2["textblob_subjectivity"] = df_2['clean'].apply(fetch_subjective_blob)

#df_2.to_csv(path + 'df2_sentiment_analyzed.csv',header=True, index=True)

#%% df3
#df_3 = pd.read_csv(path + "MARCH24_cleaned.csv")

# check sentiment score
#df_3 ["nltk_sentiment"] = df_3['clean'].apply(fetch_senti)

#df_3["textblob_polarity"] = df_3['clean'].apply(fetch_polarity_blob)

#df_3["textblob_subjectivity"] = df_3['clean'].apply(fetch_subjective_blob)

#df_3.to_csv(path + 'df3_sentiment_analyzed.csv',header=True, index=True)




#%% Here the Code starts:
    
    
#%% Import data after getting sentiment score

path = '/Users/freda/Desktop/Final/NLP/cleaned data/'
df_1 = pd.read_csv(path + "df1_sentiment_analyzed.csv")

df_2 = pd.read_csv(path + "df2_sentiment_analyzed.csv")



df_3 = pd.read_csv(path + "df3_sentiment_analyzed.csv", lineterminator='\n')


#%% correlation between textblob and nltk sentiment 
#correlation_d1 = np.corrcoef(df_1["textblob_polarity"],df_1['nltk_sentiment'] )

#correlation_d2 = np.corrcoef(df_2["textblob_polarity"],df_2['nltk_sentiment'] )

#correlation_d3 = np.corrcoef(df_3["textblob_polarity"],df_3['nltk_sentiment'] )

#%% Clean hashtag column
def clean_txt(txt_in):
    import re
    clean_str = re.sub("[^A-Za-z]+", " ", txt_in).strip().lower()
    return clean_str 

def clean_txt2(txt_in):
    import re
    clean_str = re.sub("[^A-Za-z]+", " ", str(txt_in)).strip().lower()
    return clean_str 

def delete_words(var_in):
    sw = ['text','indices']
    tmp = [word for word in var_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

df_1['clean_hashtags'] = df_1['hashtags'].apply(clean_txt).apply(delete_words)
df_2['clean_hashtags'] = df_2['hashtags'].apply(clean_txt2).apply(delete_words)
df_2_na['clean_hashtags'] = df_2_na['hashtags'].apply(clean_txt2).apply(delete_words)
df_3['clean_hashtags'] = df_3['hashtags'].apply(clean_txt).apply(delete_words)

#%% Get top hashtag df_1
from collections import Counter
all_hashtags_1 = list(df_1['clean_hashtags'])
all_hashtags_1 = ' '.join(all_hashtags_1).split()
fre_1 = Counter(all_hashtags_1)   
print(fre_1.most_common(20))

#%% Get top hashtag df_2
all_hashtags_2 = list(df_2['clean_hashtags'])
all_hashtags_2 = ' '.join(all_hashtags_2).split()
fre_2 = Counter(all_hashtags_2)   
print(fre_2.most_common(20))

all_hashtags_2na = list(df_2_na['clean_hashtags'])
all_hashtags_2na = ' '.join(all_hashtags_2na).split()
fre_2na = Counter(all_hashtags_2na)   
print(fre_2na.most_common(20))



#%% Get top hashtag df_3
all_hashtags_3 = list(df_3['clean_hashtags'])
all_hashtags_3 = ' '.join(all_hashtags_3).split()
fre_3 = Counter(all_hashtags_3)    
print(fre_3.most_common(20))

#%%





#%% get_sentiment_class function
def get_sentiment_class(var):
    if var < 0:
        return "negative"
    if var > 0:
        return "positive"
    else:
        return "neutral"

#%% get sentiment class 
df_1['nltk_sentiment_class'] = df_1.nltk_sentiment.apply(get_sentiment_class)
df_2['nltk_sentiment_class'] = df_2.nltk_sentiment.apply(get_sentiment_class)
df_3['nltk_sentiment_class'] = df_3.nltk_sentiment.apply(get_sentiment_class)

#%% average score for each dataset
# nltk sentiment 
avg_nltk_d1 = np.average(df_1['nltk_sentiment'])
avg_nltk_d2 = np.average(df_2['nltk_sentiment'])
avg_nltk_d3 = np.average(df_3['nltk_sentiment'])

# textblob polarity
avg_polar_d1 = np.average(df_1['textblob_polarity'])
avg_polar_d2 = np.average(df_2['textblob_polarity'])
avg_ploar_d3 = np.average(df_3['textblob_polarity'])

# textblob subjectivity
avg_sub_d1 = np.average(df_1['textblob_subjectivity'])
avg_sub_d2 = np.average(df_2['textblob_subjectivity'])
avg_sub_d3 = np.average(df_3['textblob_subjectivity'])

#%%




#%% variance for each day
var_d1_p = statistics.variance(df_1[df_1['nltk_sentiment_class'] == "positive"]['nltk_sentiment'])
var_d2_p = statistics.variance(df_2[df_2['nltk_sentiment_class'] == "positive"]['nltk_sentiment'])
var_d3_p = statistics.variance(df_3[df_3['nltk_sentiment_class'] == "positive"]['nltk_sentiment'])

var_d1_n = statistics.variance(df_1[df_1['nltk_sentiment_class'] == "negative"]['nltk_sentiment'])
var_d2_n = statistics.variance(df_2[df_2['nltk_sentiment_class'] == "negative"]['nltk_sentiment'])
var_d3_n = statistics.variance(df_3[df_3['nltk_sentiment_class'] == "negative"]['nltk_sentiment'])

#%% Most retweet texts

most_retweet_text1 = df_1[['retweetcount','text']].sort_values(by = 'retweetcount', ascending = False).iloc[0]
print(most_retweet_text1.text)

most_retweet_text2 = df_2[['retweetcount','text']].sort_values(by = 'retweetcount', ascending = False).iloc[0]
print(most_retweet_text2.text)

most_retweet_text3 = df_3[['retweetcount','text']].sort_values(by = 'retweetcount', ascending = False).iloc[0]
print(most_retweet_text3.text)

#%%





#%% Remove dulpicate tweets
df_1_no_duplicate = df_1[['username','location', 'totaltweets', 'retweetcount','text', 'hashtags', 'language', 'clean','nltk_sentiment', 'textblob_polarity','textblob_subjectivity','nltk_sentiment_class']]
df_1_no_duplicate.drop_duplicates(subset='text',inplace=True)

df_2_no_duplicate = df_2[['username','location', 'totaltweets', 'retweetcount','text', 'hashtags', 'language', 'clean','nltk_sentiment', 'textblob_polarity','textblob_subjectivity','nltk_sentiment_class']]
df_2_no_duplicate.drop_duplicates(subset='text',inplace=True)

df_3_no_duplicate = df_3[['username','location', 'totaltweets', 'retweetcount','text', 'hashtags', 'language', 'clean','nltk_sentiment', 'textblob_polarity','textblob_subjectivity','nltk_sentiment_class']]
df_3_no_duplicate.drop_duplicates(subset='text',inplace=True)


#%% visualization  df1 -- Pie Chart
import matplotlib.pyplot as plt

labels = "Positive", "Negative", "Neutral"

pos_count = (df_1['nltk_sentiment_class'] == "positive").sum()
neg_count = (df_1['nltk_sentiment_class'] == "negative").sum()
neu_count = (df_1['nltk_sentiment_class'] == "neutral").sum()

sizes = (pos_count, neg_count, neu_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
plt.legend()
plt.show()

#plt.bar(labels, sizes)
#plt.show()

#%% visualization  df2
# pos_count = (df_2['nltk_sentiment_class'] == "positive").sum()
# neg_count = (df_2['nltk_sentiment_class'] == "negative").sum()
# neu_count = (df_2['nltk_sentiment_class'] == "neutral").sum()

# sizes = (pos_count, neg_count, neu_count)

# plt.figure(figsize=(5,5))
# plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
# plt.legend()
# plt.show()


## HERE DROP DF2 N/A
df_2_na = df_2.dropna(subset=['clean'])

#USE df2_na data
pos_count = (df_2_na['nltk_sentiment_class'] == "positive").sum()
neg_count = (df_2_na['nltk_sentiment_class'] == "negative").sum()
neu_count = (df_2_na['nltk_sentiment_class'] == "neutral").sum()

sizes = (pos_count, neg_count, neu_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
plt.legend()
plt.show()




#%% visualization  df3
pos_count = (df_3['nltk_sentiment_class'] == "positive").sum()
neg_count = (df_3['nltk_sentiment_class'] == "negative").sum()
neu_count = (df_3['nltk_sentiment_class'] == "neutral").sum()

sizes = (pos_count, neg_count, neu_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
plt.legend()
plt.show()

#%% Also draw textblob polarity to see the diff

#df_1['textblob_sentiment_class'] = df_1.textblob_polarity.apply(get_sentiment_class)

#pos_count = (df_1['textblob_sentiment_class'] == "positive").sum()
#neg_count = (df_1['textblob_sentiment_class'] == "negative").sum()
#neu_count = (df_1['textblob_sentiment_class'] == "neutral").sum()

#sizes = (pos_count, neg_count, neu_count)

#plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
#plt.legend()
#plt.show()

#%% visualization  df2

#df_2['textblob_sentiment_class'] = df_2.textblob_polarity.apply(get_sentiment_class)

#pos_count = (df_2['textblob_sentiment_class'] == "positive").sum()
#neg_count = (df_2['textblob_sentiment_class'] == "negative").sum()
#neu_count = (df_2['textblob_sentiment_class'] == "neutral").sum()

#sizes = (pos_count, neg_count, neu_count)

#plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
#plt.legend()
#plt.show()

#%% visualization  df3

#df_3['textblob_sentiment_class'] = df_3.textblob_polarity.apply(get_sentiment_class)

#pos_count = (df_3['textblob_sentiment_class'] == "positive").sum()
#neg_count = (df_3['textblob_sentiment_class'] == "negative").sum()
#neu_count = (df_3['textblob_sentiment_class'] == "neutral").sum()

#sizes = (pos_count, neg_count, neu_count)

#plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff','#99ff99'])
#plt.legend()
#plt.show()

#%%

#%% Top 10 most positive tweets

sorted_df_1_p = df_1_no_duplicate.sort_values(by = 'nltk_sentiment', ascending = False)
print(sorted_df_1_p['text'].head(10))

sorted_df_2_p = df_2_no_duplicate.sort_values(by = 'nltk_sentiment', ascending = False)
print(sorted_df_2_p['text'].head(10))

sorted_df_3_p = df_3_no_duplicate.sort_values(by = 'nltk_sentiment', ascending = False)
print(sorted_df_3_p['text'].head(10))

#%% Top 10 most negative tweets
sorted_df_1_n = df_1_no_duplicate.sort_values(by = 'nltk_sentiment')
print(sorted_df_1_n['text'].head(10))

sorted_df_2_n = df_2_no_duplicate.sort_values(by = 'nltk_sentiment')
print(sorted_df_2_n['text'].head(10))

sorted_df_3_n = df_3_no_duplicate.sort_values(by = 'nltk_sentiment')
print(sorted_df_2_n['text'].head(10))

#%% Wordcloud for positive tweets df1
from wordcloud import WordCloud, STOPWORDS

def my_stop_words(var_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    tmp = [word for word in var_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

df_1_p = df_1_no_duplicate[df_1_no_duplicate.nltk_sentiment_class != "negative"]
df_1_p = df_1_p[df_1_p.nltk_sentiment_class != "neutral"]

text1_p = " ".join(i for i in df_1_p.text)
stopwords = set(STOPWORDS)
stopwords.add('https')
stopwords.add('t')
stopwords.add('co')
wordcloud_1_p = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text1_p))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_1_p, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for positive tweets df2
df_2_p = df_2_no_duplicate[df_2_no_duplicate.nltk_sentiment_class != "negative"]
df_2_p = df_2_p[df_2_p.nltk_sentiment_class != "neutral"]
df_2_p = df_2_p.dropna(subset = ['text'])

text2_p = " ".join(i for i in df_2_p.text)
wordcloud_2_p = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text2_p))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_2_p, interpolation='bilinear')
plt.axis("off")
plt.show()


#%% Wordcloud for positive tweets df3
df_3_p = df_3_no_duplicate[df_3_no_duplicate.nltk_sentiment_class != "negative"]
df_3_p = df_3_p[df_3_p.nltk_sentiment_class != "neutral"]

text3_p = " ".join(i for i in df_3_p.text)
wordcloud_3_p = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text3_p))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_3_p, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for negative tweets df1
df_1_n = df_1_no_duplicate[df_1_no_duplicate.nltk_sentiment_class != "positive"]
df_1_n = df_1_n[df_1_n.nltk_sentiment_class != "neutral"]

text1_n = " ".join(i for i in df_1_n.text)
wordcloud_1_n = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text1_n))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_1_n, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for negative tweets df2
df_2_n = df_2_no_duplicate[df_2_no_duplicate.nltk_sentiment_class != "positive"]
df_2_n = df_2_n[df_2_n.nltk_sentiment_class != "neutral"]

text2_n = " ".join(i for i in df_2_n.text)
wordcloud_2_n = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text2_n))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_2_n, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for negative tweets df3
df_3_n = df_3_no_duplicate[df_3_no_duplicate.nltk_sentiment_class != "positive"]
df_3_n = df_3_n[df_3_n.nltk_sentiment_class != "neutral"]

text3_n = " ".join(i for i in df_3_n.text)
wordcloud_3_n = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text3_n))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_3_n, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for neutral tweets df1
df_1_neu = df_1_no_duplicate[df_1_no_duplicate.nltk_sentiment_class != "positive"]
df_1_neu = df_1_neu[df_1_neu.nltk_sentiment_class != "negative"]

text1_neu = " ".join(i for i in df_1_neu.text)
wordcloud_1_neu = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text1_neu))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_1_neu, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for neutral tweets df2
df_2_neu = df_2_no_duplicate[df_2_no_duplicate.nltk_sentiment_class != "positive"]
df_2_neu = df_2_neu[df_2_neu.nltk_sentiment_class != "negative"]

text2_neu = " ".join(str(i) for i in df_2_neu.text)
wordcloud_2_neu = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text2_neu))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_2_neu, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% Wordcloud for neutral tweets df3
df_3_neu = df_3_no_duplicate[df_3_no_duplicate.nltk_sentiment_class != "positive"]
df_3_neu = df_3_neu[df_3_neu.nltk_sentiment_class != "negative"]

text3_neu = " ".join(i for i in df_3_neu.text)
wordcloud_3_neu = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                collocations = False,
                colormap = 'tab20',
                min_font_size = 10).generate(my_stop_words(text3_neu))

plt.figure(figsize=(20,15))
plt.imshow(wordcloud_3_neu, interpolation='bilinear')
plt.axis("off")
plt.show()


#%% Compare US to the other country df_1
df_1[['location']] = df_1[['location']].fillna('') # remove nan, na

usa_location = ['usa','USA', 'United States','nyc', 'New York', 'Los Angeles', 'Washington','Chicago','Florida','Hawaii','America','Detroit', 'New Jersey',
                'Texas', 'Boston','San Francisco', 'Houston','Phoenix','Philadelphia','San Diego','Dallas','San Jose','Austin', 'Seattle','Las Vegas',
                'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
import seaborn as sns
# histogram and a smooth estimate of the ditribution using Kernel density estimation, sns.distplot
sns.set(rc = {'figure.figsize':(20,12)})

sns.distplot(df_1[~df_1["location"].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             ## if also want to plot histogram, just set hist = True, but the graph will look a little messy
             bins=20, color = 'indianred', 
             hist_kws={'color':'salmon'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_1[df_1['location'].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'seagreen', 
             hist_kws={'color':'mediumseagreen'},
             kde_kws={'shade': True,'linewidth': 3})

df_1_us = df_1[df_1['location'].str.contains('|'.join(usa_location))]

#%% US df_2
df_2[['location']] = df_2[['location']].fillna('') # remove nan, na


sns.distplot(df_2[~df_2["location"].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'indianred', 
             hist_kws={'color':'salmon'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_2[df_2['location'].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'seagreen', 
             hist_kws={'color':'mediumseagreen'},
             kde_kws={'shade': True,'linewidth': 3})

df_2_us = df_2[df_2['location'].str.contains('|'.join(usa_location))]

#%% US df_3
df_3[['location']] = df_3[['location']].fillna('') # remove nan, na

sns.distplot(df_3[~df_3["location"].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'indianred', 
             hist_kws={'color':'salmon'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_3[df_3['location'].str.contains('|'.join(usa_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'seagreen', 
             hist_kws={'color':'mediumseagreen'},
             kde_kws={'shade': True,'linewidth': 3})

df_3_us = df_3[df_3['location'].str.contains('|'.join(usa_location))]

#%% Compare Ukraine to the other country df_1

ukraine_location = ['kyiv','ukrain','kharkiv']

df_1['location'].str.lower()
sns.distplot(df_1[~df_1["location"].str.contains('|'.join(ukraine_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'palevioletred', 
             hist_kws={'color':'pink'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_1[df_1['location'].str.contains('ukraine')]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'cornflowerblue', 
             hist_kws={'color':'lightsteelblue'},
             kde_kws={'shade': True,'linewidth': 3})

df_1_ukraine = df_1[df_1['location'].str.lower().str.contains('ukrain')]

#%% Ukraine df_2
df_2['location'].str.lower()
sns.distplot(df_2[~df_2["location"].str.contains('|'.join(ukraine_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'palevioletred', 
             hist_kws={'color':'pink'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_2[df_2['location'].str.contains('ukraine')]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'cornflowerblue', 
             hist_kws={'color':'lightsteelblue'},
             kde_kws={'shade': True,'linewidth': 3})

df_2_ukraine = df_2[df_2['location'].str.lower().str.contains('ukrain')]
#%% Ukraine df_3
df_3['location'].str.lower()
sns.distplot(df_3[~df_3["location"].str.contains('|'.join(ukraine_location))]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'palevioletred', 
             hist_kws={'color':'pink'},
             kde_kws={'shade': True,'linewidth': 3})

sns.distplot(df_3[df_3['location'].str.contains('ukraine')]["nltk_sentiment"], hist=False, kde=True, 
             bins=20, color = 'cornflowerblue', 
             hist_kws={'color':'lightsteelblue'},
             kde_kws={'shade': True,'linewidth': 3})

df_3_ukraine = df_3[df_3['location'].str.lower().str.contains('ukrain')]
#%% 






#%% 
def get_fact_class(var):
    if var < 0.3:
        return "Objective"
    else:
        return "Subjective"

#%% visualization  df1
df_1['fact_or_opinion_class'] = df_1.textblob_subjectivity.apply(get_fact_class)

labels = "Objective","Subjective"

fact_count = (df_1['fact_or_opinion_class'] == "Objective").sum()
opinion_count = (df_1['fact_or_opinion_class'] == "Subjective").sum()

sizes = (fact_count, opinion_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff'])
plt.legend()
plt.show()

#plt.bar(labels, sizes)
#plt.show()

#%% visualization df2
df_2['fact_or_opinion_class'] = df_2.textblob_subjectivity.apply(get_fact_class)

fact_count = (df_2['fact_or_opinion_class'] == "Objective").sum()
opinion_count = (df_2['fact_or_opinion_class'] == "Subjective").sum()

sizes = (fact_count, opinion_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff'])
plt.legend()
plt.show()


df_2_na['fact_or_opinion_class'] = df_2_na.textblob_subjectivity.apply(get_fact_class)

fact_count = (df_2_na['fact_or_opinion_class'] == "Objective").sum()
opinion_count = (df_2_na['fact_or_opinion_class'] == "Subjective").sum()

sizes = (fact_count, opinion_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff'])
plt.legend()
plt.show()

#%% visualization  df3
df_3['fact_or_opinion_class'] = df_3.textblob_subjectivity.apply(get_fact_class)

fact_count = (df_3['fact_or_opinion_class'] == "Objective").sum()
opinion_count = (df_3['fact_or_opinion_class'] == "Subjective").sum()

sizes = (fact_count, opinion_count)

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, colors = ['#ff9999','#66b3ff'])
plt.legend()
plt.show()


#%% HERE BEGIN




#%% Tweet Length

# Distribution of Tweet length
# df1 length


df_1['Length'] = df_1['clean'].str.split().apply(len)


import plotly.graph_objs as go
import seaborn as sns

x = df_1.Length.values

sns.distplot(x, hist=True, kde=False, 
             color = 'cornflowerblue')


# df2_na length

df_2_na['Length'] = df_2_na['clean'].str.split().apply(len)
x = df_2_na.Length.values

sns.distplot(x, hist=True, kde=False, 
             color = 'cornflowerblue')




# df3 length

df_3['Length'] = df_3['clean'].str.split().apply(len)


x = df_3.Length.values

sns.distplot(x, hist=True, kde=True, 
             color = 'cornflowerblue')


# # positive vs. negative length: Day 1 (TEST)

# df_1_pos = df_1[df_1['nltk_sentiment_class'] == 'positive']

# x = df_1_pos.Length.values

# sns.distplot(x, hist=True, kde=False, 
#              color = 'cornflowerblue')

# df_1_neg = df_1[df_1['nltk_sentiment_class'] == 'negative']

# x = df_1_neg.Length.values

# sns.distplot(x, hist=True, kde=False, 
#              color = 'cornflowerblue')


# positive aggregate
df_1_pos = df_1[df_1['nltk_sentiment_class'] == 'positive']


df_2_pos = df_2_na[df_2_na['nltk_sentiment_class'] == 'positive']


df_3_pos = df_3[df_3['nltk_sentiment_class'] == 'positive']



combined_pos = pd.concat([df_1_pos, df_2_pos, df_3_pos])

x = combined_pos.Length.values

sns.distplot(x, hist=True, kde=True, 
             color = 'cornflowerblue')

plt.axvline(x=x.mean(),color='blue')

## negative aggregate

df_1_neg = df_1[df_1['nltk_sentiment_class'] == 'negative']


df_2_neg = df_2_na[df_2_na['nltk_sentiment_class'] == 'negative']


df_3_neg = df_3[df_3['nltk_sentiment_class'] == 'negative']


combined_neg = pd.concat([df_1_neg, df_2_neg, df_3_neg])

x = combined_neg.Length.values

sns.distplot(x, hist= True, kde=True, 
             color = '#FFD500')



# statistics for combined length: positive
tmp = statistics.variance(combined_pos['Length'])

# statistics for combined length: negative
tmp2 = statistics.variance(combined_neg['Length'])


## average length by day
len_1 = np.average(df_1['Length'])
len_2 = np.average(df_2_na['Length'])
len_3 = np.average(df_3['Length'])

#%% ANOVA Test if statisticall significant

sent_d1 = df_1['nltk_sentiment'].dropna()
sent_d2 = df_2_na['nltk_sentiment'].dropna()
sent_d3 = df_3['nltk_sentiment'].dropna()
tdata = pd.DataFrame({'day_1':sent_d1, 'day_2':sent_d2, 'day_3':sent_d3})
#tdata = tdata.dropna()


tdata = pd.concat([tdata[col].sort_values().reset_index(drop=True) for col in tdata], axis=1, ignore_index=True)

tdata = tdata.rename(columns={0: "day_1",1: "day_2", 2: "day_3"})
tdata.describe()

# subjects=['id1','id2','id3','id4','id5','id6','id7']
# points = np.array(groupA +groupB + groupC)
# conditions = np.repeat(['A','B','C'],len(group0))
# subjects = np.array(subjects+subjects+subjects)
# df = pd.DataFrame({'Point':points,'Conditions':conditions,'Subjects':subjects})

tdata= tdata.dropna()
tdata.describe()

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.anova as anova

from statsmodels.stats.multicomp import pairwise_tukeyhsd


def tukey_hsd(group_names , *args ):
    endog = np.hstack(args)
    groups_list = []
    for i in range(len(args)):
        for j in range(len(args[i])):
            groups_list.append(group_names[i])
    groups = np.array(groups_list)
    res = pairwise_tukeyhsd(endog, groups)
    print (res.pvalues) #print only p-value
    print(res) #print result
    
    
print(tukey_hsd(['day_1', 'day_2', 'day_3'], tdata['day_1'], tdata['day_2'],tdata['day_3']))


# df_2_na = df_2.dropna(subset=['clean'])


