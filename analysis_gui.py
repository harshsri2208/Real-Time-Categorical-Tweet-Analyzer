import tkinter as tk
from tkinter import ttk
import analysis
from nltk.stem.porter import PorterStemmer
import train

window = tk.Tk()
 
window.title("Tweets Analysis")
window.minsize(600,400)
 
def clickMe():
    #label.configure(text= tweet.get())

    '''tweetSentiment = 0

    tweetAnalyze = tweet.get()
    tweetAnalyze = analysis.remove_pattern(tweetAnalyze, "@[\w]*")
    tweetAnalyze = ' '.join([w for w in tweetAnalyze.split() if len(w)>3])

    tweetAnalyze = tweetAnalyze.split()
    newList = []

    ps = PorterStemmer()
    for w in tweetAnalyze: 
        if newList.count(ps.stem(w)) == 0 :
            newList.append(ps.stem(w))
    tweetAnalyze = newList

    bowVector = []
    freqWords = analysis.getFreqWords()
    for x in freqWords :
        if tweetAnalyze.count(x) > 0 :
            bowVector.append(1)
        else :
            bowVector.append(0)

    print(tweetAnalyze)
    print(bowVector)

    bow = analysis.getbagOfWordsArray()
    print(len(bow))
    total_data = analysis.getTotalSet()
    print(len(total_data))
    label.configure(text = train.getSentiment(bow, analysis.getTrainSet(), bowVector))'''

    total_data = analysis.getTweet(tweet.get())
    bow = analysis.bagOfWordsArray(total_data)
    print(bow)


 
label = ttk.Label(window, text = "Enter Your Keyword")
label.grid(column = 0, row = 0)
 
 
 
 
tweet = tk.StringVar()
tweetEntered = ttk.Entry(window, width = 15, textvariable = tweet)
tweetEntered.grid(column = 0, row = 1)
 
 
button = ttk.Button(window, text = "Train Model based on query", command = clickMe)
button.grid(column= 0, row = 2)
 
window.mainloop()
