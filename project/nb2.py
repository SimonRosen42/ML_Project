from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


datapath= "issarcastic.csv"
data=pd.read_csv(datapath)

y=data.is_sarcastic
x=data

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=100)
X_train1, X_val, y_train1, y_test1= train_test_split(X_train, y_train, test_size=0.2, random_state=100)

X_train1.to_csv('sartrain.csv')
X_test.to_csv('sartest1.csv')
X_val.to_csv('sarval.csv')


with open("sartrain.csv", 'r') as file:
    reviews = list(csv.reader(file))

with open("sartest1.csv", 'r') as file:
    test = list(csv.reader(file))

with open("sarval.csv", 'r') as file:
    validation= list(csv.reader(file))

reviews.pop(0)
test.pop(0)
validation.pop(0)


vectorizer = CountVectorizer(stop_words=['is', 'a', 'i', 'my', 'we'], analyzer='word')
train_features = vectorizer.fit_transform([r[1] for r in reviews])
#val_features= vectorizer.fit_transform(r[1] for r in validation)
test_features = vectorizer.transform([r[1] for r in test])

nb = MultinomialNB(1, True, None)
nb.fit(train_features, [int(r[2]) for r in reviews])


predictions = nb.predict(test_features)
print(predictions)
actual = [int(r[2]) for r in test]
print(actual)

from sklearn.metrics import confusion_matrix

print "Confusion matrix: "
results = confusion_matrix(actual, predictions)
print(results)


fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)
print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))