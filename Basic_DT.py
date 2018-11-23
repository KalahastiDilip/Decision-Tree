# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:49:50 2018

@author: Dilip.kalahasti
"""

import os
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

import numpy as np

titanic_train=pd.read_csv("D:\\Data\\titanic_train_100.csv")

titanic_train.info()

titanic_train.describe()

titanic_train['Age'].fillna(0,inplace=True)
# to fill null values with 0
###if it is string use fillna('0',inplace=True)

titanic_train['Age']=titanic_train['Age'].mean()

X=titanic_train[['Pclass', 'SibSp', 'Parch']]
Y=titanic_train['Survived']


dt=tree.DecisionTreeClassifier(criterion = 'gini')
dt.fit(X,Y)

titanic_test = pd.read_csv("D:\\Data\\titanic_test_101_200.csv")
titanic_test['Age']=titanic_test['Age'].mean()
titanic_test.info()

X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
os.getcwd() #To get current working directory
#titanic_test.to_csv("submission_Titanic2.csv", columns=['PassengerId','Survived'], index=False)

actual=np.array(titanic_train['Survived'])
predicted=np.array(titanic_test['Survived'])

confusion_matrix(actual,predicted)
