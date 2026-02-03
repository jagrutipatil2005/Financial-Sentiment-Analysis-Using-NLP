from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import joblib 
c=CountVectorizer(ngram_range=(1,2),max_features=5000)
def preprocess_data(path):
    cleaned=pd.read_csv (path)
    x=cleaned.drop('Sentiment',axis=1)
    y=cleaned['Sentiment' ]
    l.fit(y)
    y=l.transform(y)
    return x,y
def bow(x,y):
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
    pd.DataFrame(xtest).to_csv('testing.csv')
    pd.DataFrame(ytest).to_csv('testing_answer.csv')
    c.fit(xtrain['Sentence'])
    joblib.dump(c,"count.joblib")
    xtrain=c.transform(xtrain['Sentence']).todense()
    xtrain=pd.DataFrame(np.array(xtrain))
    ytrain=pd.DataFrame(ytrain)
    xtrain.to_csv('training.csv')
    ytrain.to_csv('training_answer.csv')
x,y=preprocess_data('cleaned.csv')
bow(x,y)