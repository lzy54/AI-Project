from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line=="":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory+"Positive.txt"
    (positive_list_of_lines, positive_list_of_labels)=file_reader(file_path=positive_file, label=1)

    negative_file = source_directory+"Negative.txt"
    (negative_list_of_lines, negative_list_of_labels)=file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory+"Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels)=file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set,test_labels,trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset,label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif  probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction=-1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")
    
    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions/len(test_labels)*100,5)))

    return predictions_list, round(correct_predictions/len(test_labels)*100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram
        self.BOW=[]
        self.classCounts=[]
        self.D=0
        self.N=0
        self.labelmap={}

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def train(self, training_sentences, training_labels):
        
        # See the HW_3_How_To.pptx for details
        
        # Get number of sentences in the training set
        N_sentences = len(training_sentences)

        # This will turn the training_sentences into the format described in the HW_3_How_To.pptx
        training_set = self.word_tokenization_dataset(training_sentences)

        # Get vocabulary (dictionary) used in training set
        self.V = self.compute_vocabulary(training_set)

        # Get set of all classes
        all_classes = set(training_labels)
        #-------- TO DO (begin) --------#
        # Note that, you have to further change each sentence in training_set into a binary BOW representation, given self.V

        # Compute the conditional probabilities and priors from training data, and save them in:
        # self.prior
        # self.conditional
        # You can use any data structure you want.
        # You don't have to return anything. self.conditional and self.prior will be called in def predict():
        self.prior = {}
        self.conditional = {}
        class_word_counts = {}
        class_total_words = {}
        self.unknown_word_prob = {}
        
        # Compute the prior probability of each class
        for C in all_classes:
            self.prior[C] = 0
            self.conditional[C] = {}
            class_word_counts[C] = defaultdict(int)
            class_total_words[C] = 0
        
        for C in all_classes:
            N_C = np.sum(training_labels == C)
            self.prior[C] = N_C / N_sentences
        
        # Compute the conditional probability
        for sentence, label in zip(training_set, training_labels):
            for word in sentence:
                class_word_counts[label][word] += 1
                class_total_words[label] += 1
        
        for C in all_classes:
            self.unknown_word_prob[C] = 1 / (class_total_words[C] + len(self.V))
            for word in self.V:
                count = class_word_counts[C][word]
                self.conditional[C][word] = (count + 1) / (class_total_words[C] + len(self.V))
            

                                                 
        # -------- TO DO (end) --------#


    def predict(self, test_sentence):

        # The input is one test sentence. See the HW_3_How_To.pptx for details
        
        # Your are going to save the log probability for each class of the test sentence. See the HW_3_How_To.pptx for details
        label_probability = {
            0: 0,
            1: 0,
            -1:0,
        }

        # This will tokenize the test_sentence: test_sentence[n] will be the "n-th" word in a sentence (n starts from 0)
        test_sentence = self.word_tokenization_sentence(test_sentence)

        #-----------------------#
        #-------- TO DO (begin) --------#
        # Based on the test_sentence, please first turn it into the binary BOW representation (given self.V) and compute the log probability
        # Please then use self.prior and self.conditional to calculate the log probability for each class. See the HW_3_How_To.pptx for details 

        # Return a dictionary of log probability for each class for a given test sentence:
        # e.g., {0: -39.39854137691295, 1: -41.07638511893377, -1: -42.93948478571315}
        # Please follow the PPT to first perform log (you may use np.log) to each probability term and sum them.
        classes = self.prior.keys()
        
        for C in classes:
            log_prob_C = np.log(self.prior[C])
            for word in test_sentence:
                if word in self.V:
                    prob_w_C = self.conditional[C][word]
                else:
                    prob_w_C = self.unknown_word_prob[C]
                log_prob_C += np.log(prob_w_C)
            label_probability[C] = log_prob_C

        # -------- TO DO (end) --------#

        return label_probability

TASK = 'test'  #'train'  'test'


if TASK=='train':
    train_folder = "data-sentiment/train/"       
    training_sentences, training_labels = data_reader(train_folder)
        
    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences,training_labels)
    
    f = open('classifier.pkl', 'wb')
    pickle.dump(NBclassifier, f)
    f.close()
if TASK == 'test':
    test_folder = "data-sentiment/test/"
    test_sentences, test_labels = data_reader(test_folder)
    f = open('classifier.pkl', 'rb')
    NBclassifier = pickle.load(f)
    f.close()    
    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)

