import pandas as pd
import numpy as np
import re
import string 
from nltk.stem import PorterStemmer
porter=PorterStemmer()
punctuation=string.punctuation
p=punctuation.replace('$','').replace('%','')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
english_stopwords =stopwords.words('english')
import joblib
from sklearn.metrics import accuracy_score
c=joblib.load("count.joblib")
model=joblib.load("model.joblib")
def clean_text(x):
    a=x.lower().strip()
    pattern=r"https?://\S+|www\.\S+"
    a=re.sub(pattern,'',a)
    for i in a:
        if i in p:
            a=a.replace(i,"")
    if '%' in a:
        a=a.replace('%','percent')
    text=nltk.word_tokenize(a)
    for i in text:
        if i in english_stopwords:
            text.remove(i)
    for i in range(len(text)):
        text[i]=porter.stem(text[i])
    return " ".join(text)
def evaluate_model(test,t):
    test['Sentence']=test['Sentence'].apply(clean_text)
    test=c.transform(test["Sentence"])
    ypred=model.predict(test)
    return accuracy_score(t,ypred)
xtest=pd.read_csv('testing.csv',index_col=0)
xtest.drop('Unnamed: 0',axis=1,inplace=True)
ytest=pd.read_csv('testing_answer.csv',index_col=0)
print(evaluate_model(xtest,ytest))