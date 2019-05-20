# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:30:55 2019

@author: cpill
"""
from collections import Counter
from sklearn.metrics import confusion_matrix 
import csv
import re
import pandas as pd

with open("train1.csv", 'r') as file:
  reviews = list(csv.reader(file))


def get_Text(reviews, score):
    return " ".join([ r[1] for r in reviews if r[0]==str(score) ])

def count_Text(text):
    words = re.split("\s+", text)
    return Counter(words)

def get_y_count(score):
    return len([r for r in reviews if r[0]==str(score)])

n_text = get_Text(reviews, 0)
p_text = get_Text(reviews, 1)

n_count = count_Text(n_text)
p_count = count_Text(p_text)
# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(0)
# These are the class probabilities (we saw them in the formula as P(y)).
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)

def make_class_prediction(text, counts, class_prob, class_count): #P(text|+or-)
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob


#print("Review: {0}".format(reviews[10][1]))
#print("Negative prediction: {0}".format(make_class_prediction(reviews[10][1], n_count, prob_negative, negative_review_count)))
#print("Positive prediction: {0}".format(make_class_prediction(reviews[10][1], p_count, prob_positive, positive_review_count)))
#print(reviews)

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, n_count, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, p_count, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0
    return 1

with open("test1.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[1], make_class_prediction) for r in test]
#print(predictions)

actual = [float(r[0]) for r in test]

from sklearn import metrics

# Generate the roc curve.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))

results = confusion_matrix(actual, predictions) 
print ('Confusion Matrix :')
print(results) 

from sklearn.metrics import accuracy_score

print ("Accuracy : ", accuracy_score(actual,predictions))

from sklearn.metrics import recall_score

print("Recall: ",recall_score(actual, predictions) )

from sklearn.metrics import precision_score
print("Precision: ", precision_score(actual, predictions))

from sklearn.metrics import f1_score
print("f1_score: ", f1_score(actual,predictions))
























