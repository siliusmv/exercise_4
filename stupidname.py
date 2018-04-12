# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:33:39 2018

@author: siliusmv
"""
import numpy as np
from sklearn import tree
import sklearn
import matplotlib.pyplot as plt



# Returns classifier for some data
# using T iterations of adaBoost
def adaBoost(data, T):
    dim = data.shape
    
    x = data[0:dim[0], 2:(dim[1]-1)]     
    y = data[0:dim[0], 1]

    classifiers = []
    all_as = []
    
    weight = np.ones(dim[0]) * (1. / dim[0])

    for i in range(T):
            
        clf = tree.DecisionTreeClassifier(max_depth = 1)
        clf = clf.fit(x, y, sample_weight = weight)
        pred = clf.predict(x)
        
        epsilon = sum((pred != y) * weight)
        a = (1. / 2) * np.log((1 - epsilon) / epsilon)
        
        nw = weight * np.e**(-a * y * pred)
        nw = nw / sum(nw)
        weight = nw
        
        classifiers.append(clf)
        all_as.append(a)
        
    return(list([classifiers, all_as]))
        
# Returns predictions from an
# adaBoost algorithm
def adaPred(points, classifier):
    
    n = len(classifier[0])
    
    s = np.zeros(points.shape[0])
        
    for i in range(n):
        s = s + classifier[0][i].predict(points) * classifier[1][i]
    
    return(np.sign(s))
    
# Misclassification error    
def misClass(pred, y):
    return(sum(pred != y) / float(len(y)))


# Return an error vector for iteration 1 through max_iter
def testAdaBoost(max_iter, train_data, test_data):
    
    test_dim = test_data.shape
    x_test = test_data[0:test_dim[0], 2:(test_dim[1]-1)]
    y_test = test_data[0:test_dim[0], 1]
    errors = []
    
    for i in range(max_iter):
        classifier = adaBoost(train_data, i+1)
        pred = adaPred(x_test, classifier)
        errors.append(misClass(pred, y_test))
        
    return(errors)
        
# Loads data for task 2.3)
def getData():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split as TTS
    digits = load_digits()
    
    X_train, X_test, y_train, y_test = TTS(digits.data, digits.target, random_state=42)
    
    return(X_train, X_test, y_train, y_test)





