# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 10:50:18 2022

@author: RIZWAN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\Demo\\DataScience\\#PROJECTS\\EDA + Logistic Regression + PCA\\adult.csv\\adult.csv", encoding = 'latin-1')

print(df.info())

df[df=='?']=np.nan

for col in ['workclass', 'occupation', 'native.country']:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
print(df.isnull().sum())

X = df.drop('income', axis=1)
y= pd.DataFrame(df['income'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)

from sklearn.preprocessing import LabelEncoder
categorical = ['workclass', 'education', 'marital.status', 'occupation','relationship','race', 'sex','native.country']

for feature in categorical:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train= pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test= pd.DataFrame(scaler.transform(X_test), columns=X.columns)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
print('logistic regression accuracy score with all the feature: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

X = df.drop(['income','native.country'], axis=1)
y= df['income']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import LabelEncoder

categorical = ['workclass', 'education', 'marital.status', 'occupation','relationship','race', 'sex']
for feature in categorical:
    le = LabelEncoder()
    X_train[feature]=le.fit_transform(X_train[feature])
    X_test[feature]=le.transform(X_test[feature])
    
scaler = StandardScaler()
X_train =pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X.columns)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred1 = logreg.predict(X_test)
from sklearn.metrics import accuracy_score
print("logistic regression accuracy score with first 13 features : {0:0.4f}".format(accuracy_score(y_test, y_pred1)))

#with 12 features
X = df.drop(['income','native.country','hours.per.week'], axis=1)
y = df['income']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
categorical = ['workclass', 'education', 'marital.status', 'occupation','relationship','race', 'sex']
for feature in categorical:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])

X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X.columns)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred1 = logreg.predict(X_test)
from sklearn.metrics import accuracy_score
print("logistic regression accuracy score with first 12 features : {0:0.4f}".format(accuracy_score(y_test,y_pred1)))

#ogistic Regression with first 11 features
    
X = df.drop(['income','native.country', 'hours.per.week', 'capital.loss'], axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy score with the first 11 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    
    
    
X = df.drop(['income'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)

pca = PCA()
pca.fit(X_train)
print(pca.explained_variance_ratio_)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum>=0.90)+1
print('The number of dimensions required to preserve 90% of variance is',dim)

plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,14,1)
plt.xlabel('Number of  component')
plt.ylabel('Cumulative Explained Variance')
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    