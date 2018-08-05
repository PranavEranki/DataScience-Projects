#THIS WAS A TEST FILE FOR IMPROVING + FINALIZING BASIC CODE OUTLINE

#IMPORTS
from __future__ import print_function
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV

#Splitting train dataset
train = pd.read_csv('train.csv').values
train_y = train[:,1]
train_x = train[:,2]

#Making bag of words model
corpus_train = []
for i in range(0,train_x.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', train_x[i])
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_train.append(review)

#Count vectorizer
cv = CountVectorizer(max_features = 1500)
train_x = cv.fit_transform(corpus_train).toarray()

#Adding PCA to the model
pca = PCA(n_components=350)
train_x = pca.fit_transform(train_x)
explained_variance = pca.explained_variance_ratio_

#Fitting classifier + evaluating results
#93.8 percent, neighbors = 9
classifier = KNeighborsClassifier() 
classifier.fit(train_x,train_y)

neighbor_list = [1,3,6,9,12,15,18,21,25]
algorithm_list = ['brute', 'kd_tree', 'ball_tree']
weights_list = ['uniform', 'distance']
p_list = [1] #p_list = [1,2,3,4]
leaf_list = [10,15,20,25,30,35,40,45,50]
parameters = [{'n_neighbors':neighbor_list, 'weights':weights_list, 'algorithm':algorithm_list, 'p':p_list, 'leaf_size':leaf_list}]

'''
cv = model_selection.ShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0)
scores = model_selection.cross_val_score (classifier,train_x,train_y, cv = cv)
print(str(scores.mean()))
print(str(scores.std()))
'''
clf = GridSearchCV(classifier, parameters, n_jobs = -1)
clf = clf.fit(train_x,train_y)

print(clf.best_score_)
clf.best_params_

'''
dev_pred = classifier.predict(dev_x)
f1 = metrics.f1_score(dev_y,dev_pred)
print(f1)
'''

'''

#~~~ TEST DATA PREPROCESSING ~~~

test = pd.read_csv('test_tweets.csv').values
test = test[:,1]
test = np.reshape(test,(17197,1))

corpus_test = []
for i in range(17197):
    review = re.sub('[^a-zA-Z]', ' ', test[i][0])
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)
    
test = cv.transform(corpus_test).toarray()
test = pca.transform(test)
test = test.astype('float32')
pred = classifier.predict(test)

df = pd.DataFrame(pred)
df.index.name = 'id'
df.columns = ['label']

df.to_csv("predictions.csv")
'''