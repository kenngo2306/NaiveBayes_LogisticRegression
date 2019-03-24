This program implement Naive Bayes Learning algorithm and Logistic Regression algorithm
to classify if an email is a spam or not

Prerequisite:
1. Install python 3
2. Have training sets example for spam and not spam (ham) emails
3. Have test sets example for spam and not spam (ham) emails
4. A file contain a list of stop words to improve the algorithm 
5. Have the code - NaiveBayes_LogisticRegression.py file ready

Execution:
To run Naive Bayes learning algorithm, execute:
    python .\NaiveBayes_LogisticRegression.py "NB" <train_ham_path> <train_spam_path> <test_ham_path> <test_spam_path> <stop_words_path>
    Example:
      python .\NaiveBayes_LogisticRegression.py "NB" "./train/ham" "./train/spam" "./test/ham" "./test/spam" "./stop_words.txt"

To run Logistic Regression algorithm, execute:
	python .\NaiveBayes_LogisticRegression.py "LR" <train_ham_path> <train_spam_path> <test_ham_path> <test_spam_path> <stop_words_path> <number_of_interation> <learning_rate> <lambda_value>
	Example:	
	  python .\NaiveBayes_LogisticRegression.py "LR" "./train/ham" "./train/spam" "./test/ham" "./test/spam" "./stop_words.txt" 145 0.122 0.007

Sample output:
##### Running Logistic Regression Algorithm #####
Number of iterations =  145
Learning rate =  0.122
Lambda =  0.007
With stop words
Accuracy after training = 0.9205020920502092
###########################
Without stop words
Accuracy after training = 0.9288702928870293

##### Running Naive Bayes Learning Algorithm #####
With stop words, accuracy =  0.9079497907949791
Without stop words, accuracy =  0.9205020920502092

