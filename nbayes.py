# importing the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import re
import random
import math
import numpy as np

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
FILENAME='SMSSpamCollection'
all_data = open(FILENAME).readlines()

# split into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]


num_spam_lines = 0
num_ham_lines = 0

# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    if label == "spam":
        label = 1
        num_spam_lines += 1     # increment the number of spam lines
    else:
        label = 0
        num_ham_lines += 1      # increment the number of ham lines
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)
    
# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0

    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)


def nbayes_a():
    spam_words = []
    ham_words = []
    alpha = 0.5
    N = 20000
    for ii in range(len(train_words)):  # we pass through words in each (train) SMS
        words = train_words[ii]
        label = train_labels[ii]
        if label == 1:
            spam_words += words
        else:
            ham_words += words
    input_words = spam_words + ham_words  # all words in the input vocabulary
    
    # Count spam and ham occurances for each word
    spam_counts = {}; ham_counts = {}
    # Spamcounts
    for word in spam_words:
        try:
            word_spam_count = spam_counts.get(word)
            spam_counts[word] = word_spam_count + 1
        except:
            spam_counts[word] = 1 + alpha  # smoothening
    
    for word in ham_words:
        try:
            word_ham_count = ham_counts.get(word)
            ham_counts[word] = word_ham_count + 1
        except:
            ham_counts[word] = 1 + alpha  # smoothening
    
    num_spam = len(spam_words)
    num_ham = len(ham_words)
    
    
    # Training model starts here
    p_spam = num_spam_lines / idx_limit     # probability of spam
    p_ham = num_ham_lines / idx_limit    # probability of ham

    p_wordgivenspam = {}        # probability of each word given spam
    p_wordgivenham = {}         # probability of each word given ham
    
    denominator_spam = num_spam + (alpha * N)
    denominator_ham = num_ham + (alpha * N)
    
    for word in spam_counts:
        p_wordgivenspam[word] = (spam_counts[word] / denominator_spam)
        
    for word in ham_counts:
        p_wordgivenham[word] = (ham_counts[word] / denominator_ham)  
    # Training model ends here
        
    # Model run on test data   
    p_spamgivenline = []
    # Calculating probability of spam given the message
    for i in range(len(test_words)):
        p_spamgivenline.append(p_spam)
        for j in range(len(test_words[i])):
            if test_words[i][j] in p_wordgivenspam.keys():
                p_spamgivenline[i] = p_spamgivenline[i] * p_wordgivenspam[test_words[i][j]]
            else:
                num_spam += 1
                p_wordgivenspam[test_words[i][j]] =  alpha / denominator_spam
                p_spamgivenline[i] = p_spamgivenline[i] * p_wordgivenspam[test_words[i][j]]
            
    p_hamgivenline = []
    # Calculating probability of ham given the message
    for i in range(len(test_words)):
        p_hamgivenline.append(p_ham)
        for j in range(len(test_words[i])):
            if test_words[i][j] in p_wordgivenham.keys():
                p_hamgivenline[i] = p_hamgivenline[i] * p_wordgivenham[test_words[i][j]]
            else:
                num_ham += 1
                p_wordgivenham[test_words[i][j]] =  alpha / denominator_ham
                p_hamgivenline[i] = p_hamgivenline[i] * p_wordgivenham[test_words[i][j]]
                
    
    predicted_label = []
    # Comparing the probability of spam and ham and appending labels accordingly
    for x in range(len(p_spamgivenline)):
        if (p_hamgivenline[x] > p_spamgivenline[x]):
            predicted_label.append(0)
        else:
            predicted_label.append(1)
            
        
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0    
    # Calculating true positive and negative, false positive and negative
    for x in range(len(predicted_label)):
        if predicted_label[x] == 0 and test_labels[x] == 0:
            true_neg += 1
        elif(predicted_label[x] == 1 and test_labels[x] == 1):
            true_pos +=1
        elif(predicted_label[x] == 0 and test_labels[x] == 1):
            false_neg += 1
        else:
            false_pos += 1   
            
    print("\nSolution for Question 4.a is as follows:")
    total = true_neg + true_pos + false_neg + false_pos
    print("\nTesting Accuracy:", (true_pos + true_neg) / total, "\n")
    
    # Confusion Matrix
    data = {'Positive': pd.Series([true_pos, false_neg, ''], index = ['Positive', 'Negative', '(Predicted)']),
            'Negative': pd.Series([false_pos, true_neg, ''], index = ['Positive', 'Negative', '(Predicted)']),
            '(True)': pd.Series(['', '', ''], index = ['Positive', 'Negative', '(Predicted)'])}
    cm = pd.DataFrame(data)
    print(cm)
            
    precision = true_pos / (true_pos + false_pos)
    print("\nPrecision:", precision)
    
    recall = true_pos / (true_pos + false_neg)
    print("\nRecall:", recall)
    
    f_score = 2 * ((precision * recall) / (precision + recall))
    print("\nF-Score:", f_score)
    print("-----------------------------------------\n")
   
    
p_wordgivenspam = {}        # probability of each word given spam
p_wordgivenham = {}         # probability of each word given ham


def nbayes_b(alpha_i):
    spam_words = []
    ham_words = []
    alpha = alpha_i
    N = 20000
    for ii in range(len(train_words)):  # we pass through words in each (train) SMS
        words = train_words[ii]
        label = train_labels[ii]
        if label == 1:
            spam_words += words
        else:
            ham_words += words
    input_words = spam_words + ham_words  # all words in the input vocabulary
    
    # Count spam and ham occurances for each word
    spam_counts = {}; ham_counts = {}
    # Spamcounts
    for word in spam_words:
        try:
            word_spam_count = spam_counts.get(word)
            spam_counts[word] = word_spam_count + 1
        except:
            spam_counts[word] = 1 + alpha  # smoothening
    
    for word in ham_words:
        try:
            word_ham_count = ham_counts.get(word)
            ham_counts[word] = word_ham_count + 1
        except:
            ham_counts[word] = 1 + alpha  # smoothening
    
    num_spam = len(spam_words)
    num_ham = len(ham_words)
    
    # Training model starts here
    p_spam = num_spam_lines / idx_limit     # probability of spam
    p_ham = num_ham_lines / idx_limit    # probability of ham
 
    denominator_spam = num_spam + (alpha * N)
    denominator_ham = num_ham + (alpha * N)
    
    for word in spam_counts:
        p_wordgivenspam[word] = (spam_counts[word] / denominator_spam)
        
    for word in ham_counts:
        p_wordgivenham[word] = (ham_counts[word] / denominator_ham)
    # Training model ends here
    
    # All variables and lists that end in te denote testing variables, while the ones that end in tr are training variables.
   
    # Model run on test data
    p_spamgivenline_te = []
    # Calculating probability of spam given the message
    for i in range(len(test_words)):
        p_spamgivenline_te.append(p_spam)
        for j in range(len(test_words[i])):
            if test_words[i][j] in p_wordgivenspam.keys():
                p_spamgivenline_te[i] = p_spamgivenline_te[i] * p_wordgivenspam[test_words[i][j]]
            else:
                num_spam += 1
                p_wordgivenspam[test_words[i][j]] =  alpha / denominator_spam
                p_spamgivenline_te[i] = p_spamgivenline_te[i] * p_wordgivenspam[test_words[i][j]]
            
    p_hamgivenline_te = []
    # Calculating probability of ham given the message
    for i in range(len(test_words)):
        p_hamgivenline_te.append(p_ham)
        for j in range(len(test_words[i])):
            if test_words[i][j] in p_wordgivenham.keys():
                p_hamgivenline_te[i] = p_hamgivenline_te[i] * p_wordgivenham[test_words[i][j]]
            else:
                num_ham += 1
                p_wordgivenham[test_words[i][j]] =  alpha / denominator_ham
                p_hamgivenline_te[i] = p_hamgivenline_te[i] * p_wordgivenham[test_words[i][j]]
                
    
    predicted_label_te = []
    # Comparing the probability of spam and ham and appending labels accordingly
    for x in range(len(p_spamgivenline_te)):
        if (p_hamgivenline_te[x] > p_spamgivenline_te[x]):
            predicted_label_te.append(0)
        else:
            predicted_label_te.append(1)
    #End of test data
    
    for word in spam_counts:
        p_wordgivenspam[word] = (spam_counts[word] / denominator_spam)
        
    for word in ham_counts:
        p_wordgivenham[word] = (ham_counts[word] / denominator_ham)
        
    #Model run on training data
    p_spamgivenline_tr = []
    # Calculating probability of spam given the message
    for i in range(len(train_words)):
        p_spamgivenline_tr.append(p_spam)
        for j in range(len(train_words[i])):
            if train_words[i][j] in p_wordgivenspam.keys():
                p_spamgivenline_tr[i] = p_spamgivenline_tr[i] * p_wordgivenspam[train_words[i][j]]
            else:
                num_spam += 1
                p_wordgivenspam[train_words[i][j]] =  alpha / denominator_spam
                p_spamgivenline_tr[i] = p_spamgivenline_tr[i] * p_wordgivenspam[train_words[i][j]]
            
    p_hamgivenline_tr = []
    # Calculating probability of ham given the message
    for i in range(len(train_words)):
        p_hamgivenline_tr.append(p_ham)
        for j in range(len(train_words[i])):
            if train_words[i][j] in p_wordgivenham.keys():
                p_hamgivenline_tr[i] = p_hamgivenline_tr[i] * p_wordgivenham[train_words[i][j]]
            else:
                num_ham += 1
                p_wordgivenham[train_words[i][j]] =  alpha / denominator_ham
                p_hamgivenline_tr[i] = p_hamgivenline_tr[i] * p_wordgivenham[train_words[i][j]]
    
    predicted_label_tr = []
    # Comparing the probability of spam and ham and appending labels accordingly
    for x in range(len(p_spamgivenline_tr)):
        if (p_hamgivenline_tr[x] > p_spamgivenline_tr[x]):
            predicted_label_tr.append(0)
        else:
            predicted_label_tr.append(1)
    #End of training data 
    
    # Calculating true positive and negative, false positive and negative for test data
    true_pos_te = 0
    true_neg_te = 0
    false_pos_te = 0
    false_neg_te = 0    
    
    for x in range(len(predicted_label_te)):
        if predicted_label_te[x] == 0 and test_labels[x] == 0:
            true_neg_te += 1
        elif(predicted_label_te[x] == 1 and test_labels[x] == 1):
            true_pos_te +=1
        elif(predicted_label_te[x] == 0 and test_labels[x] == 1):
            false_neg_te += 1
        else:
            false_pos_te += 1  
            
    # Calculating true positive and negative, false positive and negative for training data
    true_pos_tr = 0
    true_neg_tr = 0
    false_pos_tr = 0
    false_neg_tr = 0    
    
    for x in range(len(predicted_label_tr)):
        if predicted_label_tr[x] == 0 and train_labels[x] == 0:
            true_neg_tr += 1
        elif(predicted_label_tr[x] == 1 and train_labels[x] == 1):
            true_pos_tr +=1
        elif(predicted_label_tr[x] == 0 and train_labels[x] == 1):
            false_neg_tr += 1
        else:
            false_pos_tr += 1  

    
    # Accuracy for test data        
    total_te = true_neg_te + true_pos_te + false_neg_te + false_pos_te
    accuracy_te = (true_pos_te + true_neg_te) / total_te
    print("\nTesting Accuracy:", accuracy_te, "\n")
    lacc_te.append(accuracy_te)
    
    # Accuracy for training data
    total_tr = true_neg_tr + true_pos_tr + false_neg_tr + false_pos_tr
    accuracy_tr = (true_pos_tr + true_neg_tr) / total_tr
    print("\nTraining Accuracy:", accuracy_tr, "\n")
    lacc_tr.append(accuracy_tr)
    
    precision_te = true_pos_te / (true_pos_te + false_pos_te)   # For testing data
    precision_tr = true_pos_tr / (true_pos_tr + false_pos_tr)   # For training data
    
    recall_te = true_pos_te / (true_pos_te + false_neg_te)  # For testing data
    recall_tr = true_pos_tr / (true_pos_tr + false_neg_tr)  # For training data

    f_score_te = 2 * ((precision_te * recall_te) / (precision_te + recall_te))
    print("\nF-Score for testing:", f_score_te) # For testing data
    lfscore_te.append(f_score_te)
    
    f_score_tr = 2 * ((precision_tr * recall_tr) / (precision_tr + recall_tr))
    print("\nF-Score for training:", f_score_tr) # For training data
    print("-----------------------------------------")
    lfscore_tr.append(f_score_tr)
    
# The program flow starts here. Here we call the nbayes_a function to execute qs 4a
nbayes_a()
print("-----------------------------------------")
# Creating all lists required for plotting graphs
lacc_te = []
lacc_tr = []
lfscore_te = []
lfscore_tr = []
print("Solution for Question 4.b is as follows: ")
# nbayes_b is called for each value of i
for i in range(-5, 1):
    nbayes_b(2 ** i) 

# plotting graph of accuracy against i
i = [-5, -4, -3, -2, -1, 0]
plt.title('Accuracy Measure')
plt.plot(i, lacc_te, label = 'Test data')
plt.plot(i, lacc_tr, label = 'Train data')
plt.xlabel('i')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting graph of F-score against i
plt.title('F-Score Measure')
plt.plot(i, lfscore_te, label = 'Test data')
plt.plot(i, lfscore_tr, label = 'Train data')
plt.xlabel('i')
plt.ylabel('F-Score')
plt.legend()
plt.show()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        