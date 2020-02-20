# Naive Bayes SMS Classification

​	The python program implements the Naive Bayes classifier. It is a type of Supervised Learning technique. A Naive Bayes classifier is based on the Bayes theorem. It has a feature matrix has assigns a class label to each data sample. Here, we assume that the features are independent.

P(A|B) = P(B|A) P(A) / P(B)



## Requirement

* Python 3
* [SMS spam collection data set](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)



## Prerequisites

* [Naive Bayes classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

  



## Running the tests

​	The program parses the input text file for the SMS data set. It then splits the data into training and testing data set. Also, it uses Naive Bayes to train the model over different alpha values and outputs the following:

* Testing Accuracy
* Confusion matrix
* Precision
* Recall
* F-score





