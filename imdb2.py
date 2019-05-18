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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('convertfull.csv')
#d = pd.read_csv('test.csv')

x_train, x_test, y_train, y_test = train_test_split(dataset['headline'], dataset['is_sarcastic'], test_size=0.2, random_state = 100)

#y_train = dataset['is_sarcastic']
#y_test = d['is_sarcastic']
   
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("[-]")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

x_train_clean = preprocess_reviews(x_train)
x_test_clean = preprocess_reviews(x_test)

vectorizer = CountVectorizer(binary=True, analyzer='word')
train_features = vectorizer.fit_transform(x_train_clean)
test_features = vectorizer.transform(x_test_clean)


X_train = pd.DataFrame(train_features.toarray())
print(X_train.shape)
X_test = pd.DataFrame(test_features.toarray())
print(X_test.shape)

class LogisticRegression:
    def __init__(self, lr=0.1, num_iter=1000, fit_intercept=True, verbose=False):
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
    
'''
model = LogisticRegression(lr=0.5, num_iter=10000)
model.fit2(X_train, y_train)
p=[]
preds = model.predict(X_test, 0.5)

for x in range(X_test.shape[0]):
    if preds[x]==True:
       p.append(1)
    else:
       p.append(0)
################PREDICTIONS##############
with open('predicted.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for a in p:
        writer.writerow([a])

csvFile.close()

prob = model.predict_prob(X_test)
############PROBABILITIES###############
with open('person1.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for b in prob:
        writer.writerow([b])

csvFile.close()

thetha = model.theta
##############THETA VALUES##############
with open('person2.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    for c in thetha:
        writer.writerow([b])

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

print ("Accuracy : ", accuracy_score(y_test,predicted)*100)

print("Confusion Matrix: ",confusion_matrix(y_test, predicted))

tp = fp = 0
# tp -> True Positive, fp -> False Positive
for i in range(0, len(predicted)-1):
    if predicted[i] == y1_test[i] == 0:
        tp = tp + 1
    elif predicted[i] == 0 and y1_test[i] == 1:
        fp = fp + 1
precision = tp/(tp + fp)

fn = 0
# fn -> False Negatives
for i in range(0, len(predicted)-1):
    if predicted[i] == 1 and y1_test[i] == 0:
        fn = fn + 1
recall = tp/(tp + fn)

tn = 0
# tn -> True Negative
for i in range(0, len(predicted)-1):
    if predicted[i] == y1_test[i] == 1:
        tn = tn + 1
        
print("Precision: ",precision)

import matplotlib.pyplot as plt
import itertools

cm = np.array([[tp, fn], [fp, tn]])

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.figure()
    else:
        print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.figure()

# Un-Normalized Confusion Matrix...
plot_confusion_matrix(cm, classes=[0,1], normalize=False, title='Unnormalized Confusion Matrix')
# Normalized Confusion Matrix...
plot_confusion_matrix(cm, classes=[0,1], normalize=True, title='Normalized Confusion Matrix')
