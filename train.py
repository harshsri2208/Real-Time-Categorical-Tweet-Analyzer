# import classes for logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

def getSentiment(bow, train, bowVector) :
    # splitting combined data

    train_bow = bow[:31962,:]
    test_bow = bow[31962:,:]

    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

    lreg = LogisticRegression() # creating an object of Logistic regression class
    lreg.fit(xtrain_bow, ytrain)

    bowVector = np.reshape(bowVector, (-1, 1000))
    prediction = lreg.predict_proba(bowVector)
    if prediction[:, 1] >= 0.3 :
        return "positive"
    return "negative"
