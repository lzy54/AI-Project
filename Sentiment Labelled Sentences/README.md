# Sentiment Labelled Sentences - Document-Term Matrix Construction

## Introduction
This project processes a dataset of sentiment-labelled sentences to construct a feature vector for each sentence using word frequency. The main objective is to create a document-term matrix where each element represents the count of a specific word in a particular sentence.

## Dataset
The dataset used is the Sentiment Labelled Sentences Data Set, which contains sentences labelled with positive or negative sentiment. The data is sourced from three platforms:

	1. IMDB
	2. Amazon
	3. Yelp

All three subsets are merged into a single file named Dataset.txt, containing a total of 3,000 sentences.

## Data Format
Each line in Dataset.txt represents a sentence and its sentiment label, separated by a tab or space. The format is:

    Sentence + [tab] + label(0: Negative; 1: Positive)
Example:

    This bar exceeded my expectations.    1
    The movie was boring and too long.    0

## Objective

	•	Tokenization: Segment each sentence into individual words.
	•	Punctuation Removal: Exclude punctuation marks from the tokens.
	•	Feature Vector Construction: Use the frequency of words in each sentence to build a document-term matrix D.
	•	D[i, j] represents the count of word j in sentence i.
	•	Output: Save the document-term matrix, labels, and vocabulary to files for future use.
## Requirements

	•	Python 3.x
	•	Libraries:
	•	NumPy
	•	pandas
	•	NLTK (Natural Language Toolkit)
	•	Punkt Tokenizer Models (nltk.download('punkt'))

## Setup

Install the Required Libraries: 

    pip install numpy pandas nltk

In your Python environment, download the Punkt tokenizer models (Alternatively, include this line in your script.):

    import nltk
    nltk.download('punkt')

Ensure that Dataset.txt is placed in the same directory as your script or adjust the file path accordingly.

## Instructions
You can change the file name in HW2.py (line 10), and output file name in HW2.py(line 65). I use 111.txt as default for saving grader's time. You can change the name to Dataset.txt for processing all 3000 sentences.

- I recommand running HW2.ipynb rather HW2.py

1.Run the Script:
   
    python script_name.py
2.Script Overview:

	•	Reads Dataset.txt and extracts sentences and labels.
	•	Tokenizes each sentence into words while removing punctuation.
	•	Builds a vocabulary of unique words.
	•	Constructs the document-term matrix D.
	•	Saves the matrix, labels, and vocabulary to files.
3.Example Use:
    
dataset:

    If you are Razr owner...you must have this!	1
    Needless to say, I wasted my money.	0
    What a waste of money and time!.	0
    And the sound quality is great.	1

output:
| if | you | are | razr | owner | ... | must | have | this | needless | to | say | i | wasted | my | money | what | a | waste | of | and | time | the | sound | quality | is | great | label |
|----|-----|-----|------|-------|-----|------|------|------|----------|----|-----|---|--------|----|-------|------|---|-------|----|-----|------|------|-------|---------|----|-------|-------|
| 1  | 2   | 1   | 1    | 1     | 1   | 1    | 1    | 1    | 0        | 0  | 0   | 0 | 0      | 0  | 0     | 0    | 0 | 0     | 0  | 0   | 0    | 0    | 0     | 0       | 0  | 0     | 1     |
| 0  | 0   | 0   | 0    | 0     | 0   | 0    | 0    | 0    | 1        | 1  | 1   | 1 | 1      | 1  | 1     | 1    | 0 | 0     | 0  | 0   | 0    | 0    | 0     | 0       | 0  | 0     | 0     |
| 0  | 0   | 0   | 0    | 0     | 0   | 0    | 0    | 0    | 0        | 0  | 0   | 0 | 0      | 0  | 0     | 1    | 1 | 1     | 1  | 1   | 1    | 1    | 1     | 0       | 0  | 0     | 0     |
| 0  | 0   | 0   | 0    | 0     | 0   | 0    | 0    | 0    | 0        | 0  | 0   | 0 | 0      | 0  | 0     | 0    | 0 | 0     | 0  | 0   | 0    | 1    | 0     | 1       | 1  | 1     | 1     |




## Notes

    •	No Stemming: The script does not perform stemming; words like “likes” and “liked” are treated as separate tokens.
	•	Punctuation Exclusion: Some punctuation marks are removed from tokens and are not included in the vocabulary or matrix.
	•	Case Normalization: All tokens are converted to lowercase for consistency.

## Troubleshooting

	•	NLTK Errors: If you encounter errors related to tokenization, ensure that the NLTK data files are downloaded by running nltk.download('punkt').
	•	File Paths: Verify that Dataset.txt is in the correct directory or adjust the data_file path in the script.
	•	Memory Issues: For large datasets, consider using sparse matrices from the scipy.sparse library to reduce memory usage.

## Acknowledgements

	•	UCI Machine Learning Repository: For providing the Sentiment Labelled Sentences Data Set.
	•	NLTK: For the natural language processing tools.
	•	NumPy and pandas: For efficient data manipulation and analysis.
