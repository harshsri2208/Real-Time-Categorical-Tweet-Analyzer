import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings
from tkinter import *
from nltk.stem.porter import * 
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%matplotlib inline

# loading training and test data

train  = pd.read_csv('Dataset/train_tweets.csv')
test = pd.read_csv('Dataset/test_tweets.csv')

#combining training and and test data for preprocessing 
total_data = train.append(test, ignore_index=True)

#function to remove @word pattern from the tweets as they do not add any value

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

# removing @username patterns from tweets using remove_pattern function

print('\n\nRemoving  Twitter Handles \n\n')
total_data['tidy_tweet'] = np.vectorize(remove_pattern)(total_data['tweet'], "@[\w]*")
total_data.head()

# words with small lenght i.e., words having length smaller than 3 hardly hold any sentiment
# hence it is better to remove such words

print('\n\nRemoving Short Words\n\n')
total_data['tidy_tweet'] = total_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
total_data.head()

# separating each word as a token

print('\n\nTweet Tokenization\n\n')
tokenized_tweet = total_data['tidy_tweet'].apply(lambda x: x.split())

'''
# removing punctuations like period, comma and semi-colon seems like a good idea but it is actually decreasing the accuracy 

#removing punctuations from each word if any
len(tokenized_tweet)
for i in range(len(tokenized_tweet)):
    for j in range(len(tokenized_tweet[i])):
        tokenized_tweet[i][j]=tokenized_tweet[i][j].replace('[.,;:]','')
        
'''        

tokenized_tweet.head()

# stemming words i.e, words are play,playing,played are treated similarly

print('\n\nStemming\n\n')
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet.head()

#stiching these tokens together

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
total_data['tidy_tweet'] = tokenized_tweet
total_data.head()

#nltk.download('punkt')

# converting tidy_tweets column to numerical value using bag of words algorithm

'''
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000, stop_words='english')

# bag-of-words feature matrix

bow = bow_vectorizer.fit_transform(total_data['tidy_tweet'])

bow=bow.toarray()
print(bow.shape)

# get an idea of bow array as it is difficult to visualise in normal form
#hence converting it to a numpy array
'''
# implementing bag of words

word2count = {}
wordlist=[]
for data in total_data['tidy_tweet'].values: 
    words = nltk.word_tokenize(data)
    wordlist.append(words)
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1
            
import heapq 
freq_words = heapq.nlargest(1000, word2count, key=word2count.get)

bow = [] 
for data,lword in zip(total_data['tidy_tweet'].values,wordlist): 
    vector = [] 
    for word in freq_words: 
        if word in lword: 
            vector.append(1) 
        else: 
            vector.append(0) 
    bow.append(vector) 
bow = np.asarray(bow)

def getbagOfWordsArray() :
    return bow

def getTrainSet() :
    return train

def getTestSet() :
    return test

def getTotalSet() :
    return total_data

def getFreqWords() :
    return freq_words