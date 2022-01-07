#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer

import warnings 
warnings.filterwarnings('ignore')


# In[172]:


df = pd.read_csv("/Users/ideakadikoy/Desktop/cses4_cut.csv")


# In[173]:


df.drop(["Unnamed: 0"], axis=1, inplace=True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
df.head()


# In[146]:


df.info()


# In[147]:


df["voted"].value_counts()


# In[148]:


df.corr().loc['voted']


# In[149]:


X=df.drop(['voted','age'],axis=1)
Y=df['voted']
age=df['age']


# In[150]:


x=X.replace([7,8,9,97,98,99,997,998,999,9997,9998,9999],np.NaN)
x=pd.concat([x,age],axis=1) 
print(x)


# In[151]:


cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0) #randomly splitting the data into test and train test. 


# # Random Forest Classifier
# 

# In[174]:


QDA = QuadraticDiscriminantAnalysis()
QDA_accuracy=cross_val_score(QDA, X, y, cv=cv).mean()

LR = LogisticRegression()
LR_accuracy=cross_val_score(LR, X, y, cv=cv).mean()

random_forest = RandomForestClassifier()
RF_accuracy=cross_val_score(random_forest, X, y, cv=cv).mean()

KNN = KNeighborsClassifier()
KNN_accuracy=cross_val_score(KNN, X, y, cv=cv).mean()


LDA = LinearDiscriminantAnalysis()
LDA_accuracy=cross_val_score(LDA, X, y, cv=cv).mean()

naive_bayes = GaussianNB()
NB_accuracy=cross_val_score(naive_bayes, X, y, cv=cv).mean()


# In[175]:




pd.options.display.float_format = '{:,.2f}%'.format
accuracies1 = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'K-Nearest Neighbors', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest','Bayes'],
    'Accuracy'    : [100*LR_accuracy, 100*KNN_accuracy, 100*LDA_accuracy, 100*QDA_accuracy, 100*RF_accuracy, 100*NB_accuracy],
    }, columns = ['Model', 'Accuracy'])

accuracies1.sort_values(by='Accuracy', ascending=False)


# # Train-Test

# In[154]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Selected top 3 classifiers

# # GaussianNB Classifier

# In[155]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
predictLR = lr.predict(X_test)


# In[156]:


accuracy_score(y_test,predictGNB) 


# In[157]:


print(confusion_matrix(y_test,predictGNB))
print("\n")
print(classification_report(y_test,predictGNB))


# In[158]:


np.set_printoptions(suppress=True) 
matNB = confusion_matrix(y_test, predictGNB) 
sns.heatmap(matNB, square=True, annot=True, cbar=False,fmt="g"); 
plt.xlabel('Predicted value');
plt.ylabel('True value');


# In[159]:


np.mean(cross_val_score(model, X, y, cv=5))


# # Random Forest Classifier

# In[160]:


model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
predictRF = model.predict(X_test)


# In[161]:


accuracy_score(y_test,predictRF)


# In[162]:


np.mean(cross_val_score(model, X, y, cv=5))


# In[163]:


print(confusion_matrix(y_test,predictRF))
print("\n")
print(classification_report(y_test,predictRF))


# In[168]:


np.set_printoptions(suppress=True) 
matNB = confusion_matrix(y_test, predictRF) 
sns.heatmap(matNB, square=True, annot=True, cbar=False,fmt="g"); 
plt.xlabel('Predicted value');
plt.ylabel('True value');


# # KNeighborsClassifier

# In[164]:


from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier()

KNE = KNeighborsClassifier()
KNE.fit(X_train,y_train)
predictKNE = lr.predict(X_test)


# In[165]:


accuracy_score(y_test,predictKNE) 


# In[166]:


print(confusion_matrix(y_test,predictKNE))
print("\n")
print(classification_report(y_test,predictKNE))


# In[167]:


np.set_printoptions(suppress=True)  
matKNE = confusion_matrix(y_test, predictKNE) #visualizing the KNeighbors confusion matrix
sns.heatmap(matKNE, square=True, annot=True, cbar=False,fmt="g");
plt.xlabel('Predicted value');
plt.ylabel('True value');

