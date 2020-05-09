import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings
import tweepy
import datetime
from time import sleep  # Used for limiting API calls
import pickle
from textblob import TextBlob
from collections import OrderedDict
import secrets
from nltk.stem.porter import * 


warnings.filterwarnings("ignore", category=DeprecationWarning)


#%matplotlib inline

#Preprocessing for sentiment analysis
def clean_tweet(tweet): 
    ''' 
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
  
def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(clean_tweet(tweet)) 
    # set sentiment 
    if analysis.sentiment.polarity < 0: 
        return 1
    else: 
        return 0

# get tweets based on a query
def getTweet(query):
    Config= secrets.twitterConfig()

    consumer_key = Config.getConsumerKey()
    consumer_secret = Config.getConsumerSecret()
    access_token = Config.getAccessToken()
    access_token_secret = Config.getAccessTokenSecret()

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    tweet_count=1000
    maxId=0
    tweets_dict={"text":[],"sentiment":[]}
    try:
        for i in range((int)(tweet_count/100)):
            fetched_tweets = api.search(q = query, count = 100,max_id=maxId) 
            maxId=fetched_tweets.max_id
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet={}
                if not tweet.text:
                    continue
               # empty dictionary to store required params of a tweet 

                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = get_tweet_sentiment(tweet.text) 

                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet["text"] not in tweets_dict["text"]: 
                        tweets_dict["text"].append(parsed_tweet["text"]) 
                        tweets_dict["sentiment"].append(parsed_tweet["sentiment"]) 
                else: 
                    tweets_dict["text"].append(parsed_tweet["text"]) 
                    tweets_dict["sentiment"].append(parsed_tweet["sentiment"])
    except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e))

    tweet_data = pd.DataFrame(tweets_dict)
    tweet_data.to_csv('Dataset/query_tweets.csv')



# loading training and test data

train  = pd.read_csv('Dataset/train_tweets.csv')
test  = pd.read_csv('Dataset/test_tweets.csv')

#combining training and and test data for preprocessing 
total_data = train

#Generate modules for the task
def preprocess():
    removePattern()
    removeShortWords()
    tokenized_tweet = tokenize()
    tokenized_tweet = stemWords(tokenized_tweet)
    joinTokens(tokenized_tweet)
    print("\n\nPreprocessing done\n\n")
    
def bagOfWordsArray() :
    preprocess()
    return bagOfWords()

def getTrainSet() :
    return train

def getTestSet() :
    return test

def getTotalSet() :
    return total_data

#function to remove @word pattern from the tweets as they do not add any value

def removePatternUtil(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.tweet(i, '', input_txt)
        
    return input_txt

# removing @username patterns from tweets using remove_pattern function
def removePattern():
    print('\n\nRemoving  Twitter Handles \n\n')
    total_data['tidy_tweet'] = np.vectorize(removePatternUtil)(total_data['tweet'], "@[\w]*")
    total_data.head()

# words with small lenght i.e., words having length smaller than 3 hardly hold any sentiment
# hence it is better to remove such words
def removeShortWords():
    print('\n\nRemoving Short Words\n\n')
    total_data['tidy_tweet'] = total_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    total_data.head()

# separating each word as a token
def tokenize():
    print('\n\nTweet Tokenization\n\n')
    tokenized_tweet = total_data['tidy_tweet'].apply(lambda x: x.split())
    return tokenized_tweet

'''
# removing punctuations like period, comma and semi-colon seems like a good idea but it is actually decreasing the accuracy 

#removing punctuations from each word if any
len(tokenized_tweet)
for i in range(len(tokenized_tweet)):
    for j in range(len(tokenized_tweet[i])):
        tokenized_tweet[i][j]=tokenized_tweet[i][j].replace('[.,;:]','')
        
'''        
# stemming words i.e, words are play,playing,played are treated similarly
def stemWords(tokenized_tweet):
    print('\n\nStemming\n\n')
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
    return tokenized_tweet

#stiching these tokens together
def joinTokens(tokenized_tweet):
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    total_data['tidy_tweet'] = tokenized_tweet
    

nltk.download('punkt')

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

def bagOfWords():
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
    return np.asarray(bow);

