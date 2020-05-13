import tkinter as tk
from tkinter import ttk
import analysis
from nltk.stem.porter import PorterStemmer
import train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

total_data = None
bow = None
lreg = None

window = tk.Tk()
 
window.title("Tweets Analysis")
window.minsize(600,400)
 
def clickMe():
    #label.configure(text= tweet.get())
    global total_data
    global bow
    global lreg

    total_data = analysis.getTweet(tweet.get())
    bow = analysis.bagOfWordsArray(total_data)

    lreg = LogisticRegression() # creating an object of Logistic regression class
    lreg.fit(bow, total_data['label'])

    window.destroy()
 
label = ttk.Label(window, text = "Enter Your Keyword")
label.grid(column = 0, row = 0)
 
tweet = tk.StringVar()
tweetEntered = ttk.Entry(window, width = 15, textvariable = tweet)
tweetEntered.grid(column = 0, row = 1) 
 
button = ttk.Button(window, text = "Train Model based on query", command = clickMe)
button.grid(column= 0, row = 2)
 
window.mainloop()

window = tk.Tk()
 
window.title("Tweets Analysis")
window.minsize(600,400)


def getSentiment() :

    global total_data
    global bow
    global lreg
    
    tweetAnalyze = tweet.get()
    tweetAnalyze = analysis.removePatternUtil(tweetAnalyze, "@[\w]*")
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

    #print(tweetAnalyze)
    #print(bowVector)

    label.configure(text = train.getSentiment(lreg, bowVector))

 
label = ttk.Label(window, text = "Enter tweet or phrase for sentiment")
label.grid(column = 0, row = 0)
 
 
 
 
tweet = tk.StringVar()
tweetEntered = ttk.Entry(window, width = 15, textvariable = tweet)
tweetEntered.grid(column = 0, row = 1)
 
 
button = ttk.Button(window, text = "Get Sentiment", command = getSentiment)
button.grid(column= 0, row = 2)

 
window.mainloop()