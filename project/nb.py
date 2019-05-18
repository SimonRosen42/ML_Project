from collections import Counter
from sklearn.metrics import confusion_matrix
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

def readFile():
    b = list()
    f = open("sarcasmdataset.txt", "r")
    f1 = f.readlines()
    for x in f1:
        b.append(x)
    return b


a1 = readFile()

datapath= "/Users/kimayramnarain1/PycharmProjects/project/sarcasmdataset.txt"
data=pd.read_csv(datapath)

y=data.is_sarcastic
x=data

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)


X_train.to_csv('sartrain.csv')
X_test.to_csv('sartest1.csv')


with open("sartrain.csv", 'r') as file:
    reviews = list(csv.reader(file))

reviews.pop(0)

#for x in range(len(reviews)):
   # print reviews[x][2], "\n"

def get_Text(reviews, score):
    return " ".join([r[1] for r in reviews if r[2] == str(score)])


def count_Text(text):
    words = re.split("\s+", text)
    return Counter(words)


def get_y_count(score):
    return len([r for r in reviews if r[2] == str(score)])


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

def make_class_prediction(text, counts, class_prob, class_count):
    prediction = 1
    text_counts = Counter(re.split("\s+", text))

    for word in text_counts:
        # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
        # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
        # We also smooth the denominator counts to keep things even.
        prediction *= text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
    # Now we multiply by the probability of the class existing in the documents.
    return prediction * class_prob


# print("Review: {0}".format(reviews[10][1]))
# print("Negative prediction: {0}".format(make_class_prediction(reviews[10][1], n_count, prob_negative, negative_review_count)))
# print("Positive prediction: {0}".format(make_class_prediction(reviews[10][1], p_count, prob_positive, positive_review_count)))
# print(reviews)

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, n_count, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, p_count, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
        return 0
    return 1

with open("sartest1.csv", 'r') as file:
    test = list(csv.reader(file ))
test.pop(0)

predictions = [make_decision(r[1], make_class_prediction) for r in test]
print(predictions)

with open("sartest1.csv", 'r') as file:
    test = list(csv.reader(file ))
test.pop(0)

actual = [int(r[2])for r in test]
print(actual)


# Generate the roc curve using scikits-learn.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))

results = confusion_matrix(actual, predictions)
print ('Confusion Matrix :')
print(results)
























