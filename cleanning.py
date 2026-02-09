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
