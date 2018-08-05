print()
print("Importing")
print()
#IMPORTS
#from __future__ import print_function
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import metrics
#from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV



def getting_data(train_dataset_name, test_dataset_name):
    print()
    print("Getting the data")
    print()
    #Parameter names are self explanatory - file names for datasets
    #This assumes you are executing this code statement from inside the directory with your datasets
    train = pd.read_csv(train_dataset_name).values
    train_y = train[:,1]
    train_x = train[:,2]
    
    test = pd.read_csv(test_dataset_name).values
    test = test[:,1]
    test = np.reshape(test,(test.shape[0],1))
    
    return train_x,train_y,test



def bagOfWords(test,train_x):
    print()
    print("Creating bag of words model")
    print()
    #Creates and returns bag-of-words versions of the test and train x
    
    #Train transformations
    corpus_train = []
    for i in range(0,train_x.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', train_x[i])
        review = review.lower().split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus_train.append(review)
    
    #Test transformations
    corpus_test = []
    for i in range(0,test.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', test[i][0])
        review = review.lower().split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus_test.append(review)
        
    return corpus_train,corpus_test



def dimensionality_reduction(corpus_train,corpus_test, return_ratio, components):
    print()
    print("Performing Dimensionality Reduction")
    print()
    #CountVectorizer
    cv = CountVectorizer(max_features = 1500)
    train_x = cv.fit_transform(corpus_train).toarray()
    
    #PCA
    pca = PCA(n_components=components)
    train_x = pca.fit_transform(train_x)
    explained_variance = pca.explained_variance_ratio_
    
        
    test = cv.transform(corpus_test).toarray()
    test = pca.transform(test)
    test = test.astype('float32')
    
    if (return_ratio):
        return train_x,test, explained_variance
    else:
        return train_x,test
    
    
    
def getOptimumParameters(useDefault, train_x,train_y, print_stats):
    print()
    print("Getting optimum parameters")
    print("This optimization algorithm may take a while, so please be patient.")
    print("Please do not do other tasks while this runs.")
    print()
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    if (useDefault):
        classifier = KNeighborsClassifier(n_neighbors = 16, leaf_size = 35, weights = 'distance')
        classifier.fit(train_x,train_y)
        return classifier
    else:
        classifier = KNeighborsClassifier() 
        classifier.fit(train_x,train_y)
        
        
        #For the sake of my program I used my own parameter lists.
        #They were derived by doing a coarse to fine babysitting model approach
        #Please change them if desired for your own problem
        neighbor_list = [16,17,18,19,20]
        weights_list = ['distance']
        leaf_list = [35,40,45,50]
        parameters = [{'n_neighbors':neighbor_list, 'weights':weights_list, 'leaf_size':leaf_list}]
        
        if print_stats:
            clf = GridSearchCV(estimator=classifier, param_grid = parameters, cv=3,refit=True, error_score=0, n_jobs = -1, verbose = 25,scoring='f1')
        else:
            clf = GridSearchCV(estimator=classifier, param_grid = parameters, cv=3,refit=True, error_score=0, n_jobs = -1,scoring='f1')
        clf = clf.fit(train_x,train_y)
        
        print(clf.best_params_)
        
        return clf
    
    
    
def predictions(classifier, train_x, train_y, test, ratio):
    print()
    print("Making predictions")
    print()
    #Changing types to work with a classifier
    train_x= train_x.astype('float32')
    train_y = train_y.astype('float32')
    
    #Splitting training set into a training + dev set
    train_x,dev_x,train_y,dev_y = train_test_split(train_x,train_y,test_size = ratio, random_state=0)
    #Making predictions
    test = test.astype('float32')
    pred = classifier.predict(test)
    return pred



def convertPredToCsv(pred, csv_name):
        
    df = pd.DataFrame(pred)
    df.index.name = 'id'
    df.columns = ['label']
    df.index = df.index + 31962
    df.to_csv(csv_name)
    print("Done making predictions.")

    


def main():
    #Retrieving the data
    train_x,train_y,test = getting_data('train.csv', 'test_tweets.csv')
    #Constructing Bag of words model
    corpus_train,corpus_test = bagOfWords(test,train_x)
    #Performing Dimensionality Reduction
    train_x,test = dimensionality_reduction(corpus_train,corpus_test,False,350)
    #Getting the optimum classifier
    classifier= getOptimumParameters(True, train_x,train_y, False)
    #Predicting + converting to csv
    pred = predictions(classifier, train_x, train_y, test, 0.1)
    convertPredToCsv(pred, 'predictions.csv')
    
    
if __name__ == "__main__":
    main()