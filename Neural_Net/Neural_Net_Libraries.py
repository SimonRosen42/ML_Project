import datetime
import os

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\User\PycharmProjects\ML_Project\data\Pre-processed_Sarcasm_Headlines_Dataset.csv")

# Split training and test data
headline_train, headline_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.20, random_state=1000)
# Split training into training and validation
headline_train, headline_validate, y_train, y_validate = train_test_split(headline_train, y_train, test_size=0.20, random_state=1000)

# Use this later - ngram_range : tuple (min_n, max_n)
vectorizer = CountVectorizer()
vectorizer.fit(headline_train)

X_train = vectorizer.transform(headline_train)
X_validate = vectorizer.transform(headline_validate)
X_test = vectorizer.transform(headline_test)

# # Normalise
# tfidfTransformer = TfidfTransformer()
# X_train = tfidfTransformer.fit_transform(X_train)

#  Neural Net
# ------------
from keras.models import Sequential
from keras import layers

# Should speed up model. For tensor cores on RTX graphics cards
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(20, input_dim=input_dim, activation='relu')) # Was 10
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Fit model and record time taken to fit model
currTime = datetime.datetime.now()
history = model.fit(X_train, y_train, epochs=1, verbose=False, validation_data=(X_test, y_test), batch_size=32)
nextTime = datetime.datetime.now()
timeElapsed = nextTime - currTime
print("Time Elapsed: ", str(timeElapsed))

loss, accuracy = model.evaluate(X_validate, y_validate, verbose=False)
print("Validate Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Confusion Matrices
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
sklearn.metrics.accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ")
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Normalised Confusion Matrix: ")
print(cm)
