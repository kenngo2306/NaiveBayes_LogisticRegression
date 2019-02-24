# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:18:12 2019

@author: Cuong Ngo
netid cpn180001
CS6375 Machine Learning -
Naive Bayes - Logistic Regression algorithms
for text classification problem - spam detection
"""

import os 
import glob
import pandas as pd
import math

ham_train_set_path='./train/ham'
spam_train_set_path='./train/spam'

# vocabulary bag is a dictionary
#   key: the unique word
#   value: array of 2 items
#       item0: count of the unique word in ham set
#       item1: count of the unique word in spam set

# function to add new word to volabulary with 0 count
def add_new_column(vocab_dict, word):
    vocab_dict[word] = [0,0,0.0,0.0]
    return vocab_dict

def extract_vocabulary():
    vocab_dict = {}
    n_0 = 0
    # build dictionary with ham emails
    for filename in glob.glob(os.path.join(ham_train_set_path, '*.txt')):
        n_0 += 1
        with open(filename, errors='ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column(vocab_dict, word)
                    vocab_dict[word][0] += 1
    n_1 = 0
    # build dictionary with spam emails       
    for filename in glob.glob(os.path.join(spam_train_set_path, '*.txt')):
        n_1 += 1
        with open(filename, errors = 'ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column(vocab_dict, word)
                    vocab_dict[word][1] += 1

    n = n_0 + n_1

    return n, n_0, n_1, vocab_dict
    

def count_in_ham(vocab_dict):
    n_count = 0
    # build dictionary with ham emails
    for filename in glob.glob(os.path.join(ham_train_set_path, '*.txt')):
        n_count += 1
        with open(filename, errors='ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column(vocab_dict, word)
                    vocab_dict[word][0] += 1
        return n_count

def count_in_spam(vocab_dict):
    n_count = 0
    # build dictionary with spam emails       
    for filename in glob.glob(os.path.join(spam_train_set_path, '*.txt')):
        n_count += 1
        with open(filename, errors = 'ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column(vocab_dict, word)
                    vocab_dict[word][1] += 1
        return n_count

def train_multinomial_nb ():
    n, n_0, n_1, vocab = extract_vocabulary()
    
    total_words_class_0 = 0
    total_words_class_1 = 1
    for word in vocab: 
        total_words_class_0 += vocab[word][0]
        total_words_class_1 += vocab[word][1]
    
    prior = [n_0 / n, n_1 / n]
    
    for word in vocab:
        # conditional probability for each word in class 0
        vocab[word][2] = (vocab[word][0] + 1) / (total_words_class_0 + 1) 
        
        # conditional probability for each word in  class 1
        vocab[word][3] = (vocab[word][1] + 1) / (total_words_class_1 + 1) 
    
    return prior, vocab

def apply_multinomial_nb (vocab, prior, filename):
    score = [math.log10(prior[0]), math.log2(prior[1])]
    with open(filename, errors = 'ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    score[0] += math.log2(vocab[word][2])
                    score[1] += math.log2(vocab[word][1])
    return 0 if score[0] >= score[1] else 1
                    

print (test_return())   
#prior_class_0, prior_class_1, vocab = train_multinomial_nb()


        


