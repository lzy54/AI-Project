import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import string

# Download NLTK data file (only on first run)
nltk.download('punkt')
# Path to the merged data file
data_file = 'Dataset.txt'

# List of sentences and labels
sentences = []
# 0:negative or 1:positive
labels = []

# Get the data
with open(data_file, 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line into sentence and label
        sentence, label = line.split('\t', 1)
        # Append the sentence and label to the lists
        sentences.append(sentence)
        # label is 0 or 1, so we need to convert it to int
        labels.append(int(label))
        
# Tokenize the sentences and remove punctuation
tokenized_sentences = []
for sentence in sentences:
    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Append the tokenized sentence to the list
    tokenized_sentences.append(tokens)
    
# vocabulary
vocabulary = {}
for tokens in tokenized_sentences:
    for token in tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
            
# Set up the matrix
M = len(tokenized_sentences)
N = len(vocabulary)
D = np.zeros((M, N), dtype=int)

for i, tokens in enumerate(tokenized_sentences):
    for token in tokens:
        j = vocabulary[token]
        D[i,j] += 1
        
# Convert the matrix to a DataFrame
df_D = pd.DataFrame(D, columns=vocabulary.keys())
# Add the labels to the DataFrame if needed
df_D['label'] = labels

# Save the DataFrame to a CSV file
df_D.to_csv('data_matrix.csv', index=False)

print(f'Document-term matrix shape: {D.shape}')







