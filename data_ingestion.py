import pandas as pd 
import pandas as pd
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
# cleaning of data 
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

def load_data(path):
    data= pd.read_csv(path)
    data.drop_duplicates(inplace=True)
    data['Sentence']=data['Sentence'].apply(clean_text)
    return data
Cleaned_data=load_data('data.csv.csv')
Cleaned_data.to_csv('cleaned.csv')



