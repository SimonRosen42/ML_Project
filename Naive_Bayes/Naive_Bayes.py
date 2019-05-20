import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\User\PycharmProjects\ML_Project\data\Pre-processed_Sarcasm_Headlines_Dataset.csv")

headlines = df['headline']
y = df['is_sarcastic']

headlines_train, headlines_test, y_train, y_test = train_test_split(headlines, y, test_size=0.10, random_state=1000)

# Create dictionary of words from training set
vectorizer = CountVectorizer()
vectorizer.fit(headlines_train)

X_train = vectorizer.transform(headlines_train)
X_test = vectorizer.transform(headlines_test)

# Note: class1 is not sarcastic and class2 is sarcastic
# y_train_array = y_train
y_train_class1_inds = np.where(y_train==0) # Indices of all non sarcastic elements
y_train_class2_inds = np.where(y_train==1) # Indices of all sarcastic elements
X_train_class1 = X_train[y_train_class1_inds]
X_train_class2 = X_train[y_train_class2_inds]

# X_train_np = X_train.toarray()
# get shape of matrix - no of rows = no of elements
class1_shapeR, class1_shapeC = X_train_class1.shape
class2_shapeR, class2_shapeC = X_train_class2.shape

prob_class1 = class1_shapeR/(class1_shapeR+class2_shapeR)
prob_class2 = class2_shapeR/(class1_shapeR+class2_shapeR)

# Less efficient but easier to code... Might change later
X_train_class1_arr = X_train_class1.toarray()
X_train_class2_arr = X_train_class2.toarray()


