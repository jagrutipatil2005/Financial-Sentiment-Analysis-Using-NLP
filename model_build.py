from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier 
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import joblib 
knn=KNeighborsClassifier(n_neighbors=5,weights='uniform')#53
g=GaussianNB() #54
m=MultinomialNB() #68
b=BernoulliNB() #67
l=LogisticRegression(max_iter=1000)
#gr=GradientBoostingClassifier() #66
r=RandomForestClassifier()#65
#s=SVC() #66
models=[knn,g,m,l]
def train_model(xtrain,ytrain):
    stratified=StratifiedKFold(n_splits=7)
    mean_scores=[]
    for i in models:
        scores=cross_val_score(cv = stratified,X = np.array(xtrain),y = np.array(ytrain).ravel(),scoring = "accuracy",estimator = i)
        print(i)
        print(scores,scores.mean())
        mean_scores.append(scores.mean())
        print("********************************")
    mean_scores=np.array(mean_scores)
    index=np.argmax(mean_scores)
    best_model=models[index]
    best_model.fit(np.array(xtrain),np.array(ytrain).ravel())
    return best_model
xtrain=pd.read_csv('training.csv',index_col=0)
ytrain=pd.read_csv('training_answer.csv',index_col=0)
best_model=train_model(xtrain,ytrain)
joblib.dump(best_model,"model.joblib")