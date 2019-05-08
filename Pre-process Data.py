import pandas as pd
import nltk

from nltk.stem import PorterStemmer

df = pd.read_json("C:/Users/Simon/PycharmProjects/ML_Project/data/Sarcasm_Headlines_Dataset.json", lines=True)

dataHeading = "headline"
className = "is_sarcastic"

# Remove article_link column as it's not necessary
del df['article_link']

# Convert to lower case
df[dataHeading] = df[dataHeading].map(lambda x: x.lower())

# Replace '!' with 'exclamationmark' as punctuation will be removed later
df[dataHeading] = df[dataHeading].replace('!', 'exclamationmark')

#%%
# Remove punctuation
df[dataHeading] = df[dataHeading].str.replace('[^\w\s]', '')

#%%
# tokenize the headline into single words using nltk
df[dataHeading] = df[dataHeading].apply(nltk.word_tokenize)

# perform word stemming
# normalise text for all variations of words that carry the same meaning, regardless of tense
stemmer = PorterStemmer()
df[dataHeading] = df[dataHeading].apply(lambda x: [stemmer.stem(y) for y in x])

# Store processed data into csv file
file_name = "data/Pre-processed_Sarcasm_Headlines_Dataset.csv"
df.to_csv(file_name, encoding='utf-8', index=False)
