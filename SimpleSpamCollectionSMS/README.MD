# What?
This is a simple SMS Spam Classification problem.

I found it online while browsing for projects.

# Where
There are various places to download the data.

I did the entire project from start to finish in the jupyter notebook labeled 'code'. It contains markdown labeling of my code.

# Breakdown

Here is a simple breakdown of what we are going to do in this problem.

## Problem Type?
This is a NLP Binary Classification problem.

## How will we clean the data?
* First, we need to remove all punctuation from the words.
* Then, we need to create a bag of words model using stopwords.
* We can also remove capitalization just for some consistency.
* Then, we need to apply the tfidf vectorizer with this bag of words model, and filter out the top words for classification.
* Now, we can proceed with some EDA

## EDA (?)
EDA does not really apply for most NLP problems once the data is cleaned and completed (atleast from my experience). However, we can look at the data and try to understand:
* Lengths of messages
* Differences in length of messages based on spam/ham
* The most frequently used words

## What classifiers can we use?
Some different binary classifiers we can use later in our code:
* SVM
* KNN
* Multinomial NB

## Metrics(?)
What metrics can we use to evaluate our classifier?

We can use a variety of metrics, including:
* Accuracy score
* Precision score
* Recall score
* F1 score
* Classification report, which displays precision, recall, and F1

## Model Selection?
What we can do for model selection is:
* Try all the models
* Select the best base classifier
Once we have found the best base classifier:
* We can do a basic grid search with k fold validatiom
* Find the best classifier with current params
* Do a course to fine grid search, drilling down into the params over time

## Displaying results
* We can use a classification report 
* We will also display the confusion matrix
* We can also display the f1 score progress over time