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
import copy
import random

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

#prior, vocab = train_multinomial_nb([])
#total = 0
#correct = 0
#for test_ham_file_name in glob.glob(os.path.join(ham_test_set_path, '*.txt')):
#    total += 1
#    if apply_multinomial_nb(vocab, prior, test_ham_file_name) == 0:
#        correct += 1
#
#for test_spam_file_name in glob.glob(os.path.join(spam_test_set_path, '*.txt')):
#    total += 1
#    if apply_multinomial_nb(vocab, prior, test_spam_file_name) == 1:
#        correct += 1  
#
#print (correct)        
#print (total)
#print (correct/total)
#
#print ("######################")   
#prior, vocab = train_multinomial_nb(read_stop_words("./stop_words.txt"))
#total = 0
#correct = 0
#for test_ham_file_name in glob.glob(os.path.join(ham_test_set_path, '*.txt')):
#    total += 1
#    if apply_multinomial_nb(vocab, prior, test_ham_file_name) == 0:
#        correct += 1
#
#for test_spam_file_name in glob.glob(os.path.join(spam_test_set_path, '*.txt')):
#    total += 1
#    if apply_multinomial_nb(vocab, prior, test_spam_file_name) == 1:
#        correct += 1  
#
#print (correct)        
#print (total)
#print (correct/total)
    
#print(apply_multinomial_nb(vocab, prior, filename2))


############# Logistic Regression Algorithm part ###################
w0_string = 'w0_value'

# def
def add_new_column_2(vocab_dict, word):
    vocab_dict[word] = 0
    return vocab_dict
# Build a list of vectors of count of word, 1 vector per training example
    
def build_vectors(ham_set, spam_set):
    vectors = []
    # build vectoes with ham emails
    for filename in glob.glob(os.path.join(ham_set, '*.txt')):    
        vector = [{},0]   
        # count apprearance of each word for particular training example and store it in vocab_dict 
        vocab_dict = {}
        vocab_dict[w0_string] = 1
        with open(filename, errors='ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column_2(vocab_dict, word)
                    vocab_dict[word] += 1
                    
        vector[0] = vocab_dict
        vectors.append(vector)
    
    for filename in glob.glob(os.path.join(spam_set, '*.txt')):
        vector = [{},1]
        # count apprearance of each word for particular training example and store it in vocab_dict 
        vocab_dict = {}  
        vocab_dict[w0_string] = 1
        with open(filename, errors='ignore') as f:
            for line in f:
                line_arr= line.strip().split(' ')
                for word in line_arr:
                    if word not in vocab_dict:
                        add_new_column_2(vocab_dict, word)
                    vocab_dict[word] += 1
                    
        vector[0] = vocab_dict
        vectors.append(vector)

    return vectors  




# function to initialize w with a initial value
def initialized_w (word_vectors, initial_value):
    w_vector = {}
    w_vector[w0_string] = initial_value
    for vector in word_vectors:
        for word, count in vector[0].items():
            if word not in w_vector:
                w_vector[word] = initial_value
    return w_vector


# function to calculate P(Y=1 | X,W) for each training example
def calculate_P(w_vector, word_vectors):
    p_vector = []
    
    # loop through each training example
    for word_vector in word_vectors:
        my_sum = 0 
        exp_value = 0
        p_value = 0
        for word, value in word_vector[0].items():
            my_sum += 0 if word not in w_vector else w_vector[word] * value
        w0 = w_vector[w0_string] 
        
        try:
            exp_value = math.exp(w0 + my_sum)
            p_value = exp_value / (1 + exp_value)
            
        except OverflowError:
            p_value = 1
        
        p_vector.append(p_value)
        
    return p_vector
    


def update_w_vector(w_vector, p_vector, word_vectors, learning_rate, my_lambda):
    w_vector_updated = {}
    
    for w_key, w_value in w_vector.items():
         
        my_sum = 0
        
        for i, word_vector in enumerate(word_vectors):
            my_sum += 0 if (w_key not in word_vector[0]) else (word_vector[0][w_key] *  (word_vector[1] -  p_vector[i]))
            
       
        
        w_vector_updated[w_key] = w_value + learning_rate * my_sum - learning_rate * my_lambda * w_value
        
    return w_vector_updated




def logistic_regression(iterations, learning_rate, lambda_value):
    word_vectors = build_vectors(ham_train_set_path, spam_train_set_path)  
    w_vector = initialized_w(word_vectors, 1)
    
    print('accuracy before training =', calculate_accuracy(w_vector, ham_test_set_path, spam_test_set_path))
    print('learning rate= ', learning_rate) 
    print('lambda= ', lambda_value) 
    
    max_i = 0
    max_accuracy = 0
    
    for i in range(iterations): 
        
        calculate_P(w_vector, word_vectors)
        p_vector = calculate_P(w_vector, word_vectors)
        w_vector = update_w_vector(w_vector, p_vector, word_vectors, learning_rate, lambda_value)
        
        tmp_accuracy = calculate_accuracy(w_vector, ham_test_set_path, spam_test_set_path)
        if max_accuracy < tmp_accuracy:
            max_accuracy = tmp_accuracy
            max_i = i
            
#        print('accuracy training at ', i, '= ', calculate_accuracy(w_vector, ham_test_set_path, spam_test_set_path))
        
    print('accuracy max after training =', max_accuracy, ' at i=', max_i)

def calculate_accuracy(w_vector, ham_set, spam_set):
    word_vectors = build_vectors(ham_set, spam_set)
    p_vector = calculate_P(w_vector, word_vectors)
    total_examples = len(p_vector)
    correct_count = 0
    
    predicted_y_value = 0
    
    for i in range(0,len(p_vector)):
        predicted_y_value = 1 if p_vector[i] >= 0.5 else 0
        if word_vectors[i][1] == predicted_y_value:
            correct_count += 1
    
    return correct_count / total_examples    


logistic_regression(500, 0.01, 0.003)

iteration = 200

for i in range (20):
    logistic_regression(iteration, random.uniform(0.01,0.3), random.uniform(0.001, 0.03))


print ('done')
        
    


        


