# import classes for logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

def getSentiment(lreg, bowVector) :

    bowVector = np.reshape(bowVector, (-1, 1000))
    prediction = lreg.predict_proba(bowVector)
    if prediction[:, 1] >= 0.3 :
        return "positive"
    return "negative"
