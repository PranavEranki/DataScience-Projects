# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:29:03 2018

@author: PranavEranki
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

train_data = pd.read_csv("TrainingData.csv")
test_data = pd.read_csv("TestDatas.csv")

train_x = train_data.iloc[:,:5].values
train_y = train_data.iloc[:,5].values
test_x = test_data.iloc[:,:].values
'''
train_x = train_data.iloc[:450,:5].values
train_y = train_data.iloc[:450,5].values

dev_x = train_data.iloc[450:,:5].values 
dev_y = train_data.iloc[450:,5].values
'''
#Feature Scaling is necessary for PCA
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
#dev_x = sc.fit_transform(dev_x)
test_x = sc.fit_transform(test_x)
#Reshaping to necessary shape
#dev_y = dev_y.reshape((126,1))

train_y = train_y.reshape((576,1))

#Applying PCA, optimal = 3
pca = PCA(n_components = 3)
train_x = pca.fit_transform(train_x)
#dev_x = pca.transform(dev_x)

test_x = pca.transform(test_x)

#Kernel SVM with a linear kernel and 3 features:
#returns a 90.47& accuracy
#C-score is non-effective

classifier = SVC(kernel = 'poly', random_state = 0,coef0=1)
classifier.fit(train_x,train_y)
y_pred = classifier.predict(test_x)

y_pred = classifier.predict(test_x)
prediction = pd.DataFrame(y_pred, columns=['Made Donation in March 2007'])
prediction.index.name = None
prediction.rename_axis(None)
prediction = prediction.to_csv('prediction.csv')
