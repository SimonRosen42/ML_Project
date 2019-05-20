import pandas as pd
import nltk
import sklearn
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix

df = pd.read_json(r"C:\Users\User\PycharmProjects\ML_Project\data\Sarcasm_Headlines_Dataset.json", lines=True)

dataHeading = "headline"
className = "is_sarcastic"

# Remove article_link column as it's not necessary
del df['article_link']

# # Convert to lower case
# df[dataHeading] = df[dataHeading].map(lambda x: x.lower())
#
# # Replace '!' with 'exclamationmark' as punctuation will be removed later
# df[dataHeading] = df[dataHeading].replace('!', 'exclamationmark')
#
# # Remove punctuation
# df[dataHeading] = df[dataHeading].replace('[^\w\s]', '')
#
# # Remove numbers
# df[dataHeading] = df[dataHeading].replace('\d', '')
#
# sum0 = (df["is_sarcastic"]==0).sum()
# sum1 = (df["is_sarcastic"]==1).sum()
# total = sum0 + sum1

#
# # Train
# # Split training and test data
# headline_train, headline_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.10, random_state=1000)
#
# # Use this later - ngram_range : tuple (min_n, max_n)
# vectorizer = CountVectorizer()
# vectorizer.fit(headline_train)
#
# X_train = vectorizer.transform(headline_train)
# X_test = vectorizer.transform(headline_test)
#
# # Tfidf - Research what this actually does...
# # Doesn't do much for accuracy
# tfidfTransformer = TfidfTransformer()
# tfidfTransformer.ngram_range = (1,2)
# X_train_tfidf = tfidfTransformer.fit_transform(X_train)
#
# # Predict the Test set results, find accuracy
# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
#
# Making the Confusion Matrix
# y_pred = classifier.predict(X_test)
# sklearn.metrics.accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
#
# print("Accuracy: ", score)
# print("Confusion Matrix: ")
# print(cm)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print("Normalised Confusion Matrix: ")
# print(cm)