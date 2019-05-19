# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:41:41 2019

@author: cpill
"""

import numpy as np
from math import exp
import pandas as pd
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('convertfull.csv')
#d = pd.read_csv('test.csv')

x_train, x_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.2, random_state = 100)

x_Train, x_val, y_Train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state = 100)
   
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("[-]")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

x_train_clean = preprocess_reviews(x_Train)
x_val_clean = preprocess_reviews(x_val)
x_test_clean = preprocess_reviews(x_test)

vectorizer = TfidfVectorizer(analyzer='word',smooth_idf=True)
train_features = vectorizer.fit_transform(x_train_clean)
val_features = vectorizer.transform(x_val_clean)
test_features = vectorizer.transform(x_test_clean)


X_train = pd.DataFrame(train_features.toarray())
print(X_train.shape)
X_val = pd.DataFrame(val_features.toarray())
print(X_val.shape)
X_test = pd.DataFrame(test_features.toarray())
print(X_test.shape)

class LogisticRegression:
    def __init__(self, lr=0.5, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    def fit2(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 100 == 0):
                    z = np.dot(X, self.theta)
                    h = self.__sigmoid(z)
                    print(f'loss: {self.__loss(h, y)} \t')
    
'''
model = LogisticRegression(lr=0.5, num_iter=10000)
model.fit2(X_train, y_Train)
p=[]
preds = model.predict(X_test, 0.5)

for x in range(X_test.shape[0]):
    if preds[x]==True:
       p.append(1)
    else:
       p.append(0)

################PREDICTIONS##############
with open('prediction.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for a in p:
        writer.writerow([a])

csvFile.close()

'''
def getfile(filename,results):
   f = open(filename)
   filecontents = f.readlines()
   for line in filecontents:
     foo = line.rstrip()
     results.append(int(foo))
   return results

predicted = []

getfile('predicted.csv',predicted)
y1_test=np.array(y_test)     

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("Confusion Matrix: ",confusion_matrix(y_test, predicted))

print ("Accuracy : ", accuracy_score(y_test,predicted))

from sklearn.metrics import recall_score

print("Recall: ",recall_score(y_test, predicted) )

from sklearn.metrics import precision_score
print("Precision: ", precision_score(y_test, predicted))

from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test,predicted))