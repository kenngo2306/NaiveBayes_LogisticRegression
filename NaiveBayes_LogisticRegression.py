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

ham_test_set_path='./test/ham'
spam_test_set_path='./test/spam'

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

def train_multinomial_nb (stop_words):
    n, n_0, n_1, vocab = extract_vocabulary()
    for word in stop_words:
        if word in vocab:
            vocab.pop(word)
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

def custom_log2(number):
    if number > 0:
        return math.log2(number)
    else:
        return 0

def apply_multinomial_nb (vocab, prior, filename):
    score = [math.log10(prior[0]), math.log2(prior[1])]
    with open(filename, errors = 'ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word in vocab:
                        score[0] += custom_log2(vocab[word][2])
                        score[1] += custom_log2(vocab[word][3])
    return 0 if score[0] >= score[1] else 1
                    

def read_stop_words(path):
    stop_words = []
    with open(path) as f:
        for line in f:
            stop_words.append(line.strip())
    return stop_words

prior, vocab = train_multinomial_nb([])
total = 0
correct = 0
for test_ham_file_name in glob.glob(os.path.join(ham_test_set_path, '*.txt')):
    total += 1
    if apply_multinomial_nb(vocab, prior, test_ham_file_name) == 0:
        correct += 1

for test_spam_file_name in glob.glob(os.path.join(spam_test_set_path, '*.txt')):
    total += 1
    if apply_multinomial_nb(vocab, prior, test_spam_file_name) == 1:
        correct += 1  

print (correct)        
print (total)
print (correct/total)

print ("########")   
prior, vocab = train_multinomial_nb(read_stop_words("./stop_words.txt"))
total = 0
correct = 0
for test_ham_file_name in glob.glob(os.path.join(ham_test_set_path, '*.txt')):
    total += 1
    if apply_multinomial_nb(vocab, prior, test_ham_file_name) == 0:
        correct += 1

for test_spam_file_name in glob.glob(os.path.join(spam_test_set_path, '*.txt')):
    total += 1
    if apply_multinomial_nb(vocab, prior, test_spam_file_name) == 1:
        correct += 1  

print (correct)        
print (total)
print (correct/total)
    
#print(apply_multinomial_nb(vocab, prior, filename2))


        


