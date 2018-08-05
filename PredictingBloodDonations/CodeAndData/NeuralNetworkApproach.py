# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv("TrainingData.csv")
test_data = pd.read_csv("TestDatas.csv")

train_x = train_data.iloc[:450,:5].values
train_y = train_data.iloc[:450,5].values

dev_x = train_data.iloc[450:,:5].values 
dev_y = train_data.iloc[450:,5].values
#Feature Scaling is necessary for PCA
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
dev_x = sc.fit_transform(dev_x)

dev_y = dev_y.reshape((126,1))
train_y = train_y.reshape((450,1))

#Applying PCA
pca = PCA(n_components = 3)
#Two components found with minimal improvement, removed
train_x = pca.fit_transform(train_x)
dev_x = pca.transform(dev_x)
explained_variance = pca.explained_variance_ratio_


#Basic NN architecture from sklearn
from sklearn.neural_network import MLPClassifier  
mlp = MLPClassifier(hidden_layer_sizes=(6, 6, 6), max_iter=1000)  
mlp.fit(train_x, train_y) 

predictions = mlp.predict(dev_x)

print (confusion_matrix(dev_y,predictions))