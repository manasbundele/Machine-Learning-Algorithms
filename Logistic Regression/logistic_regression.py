#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/park_train.data", header=None)
df_validation = pd.read_csv("/Users/manasbundele/Downloads/park_validation.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/park_test.data", header=None)

X = df_train.iloc[:,1:].copy()
X2 = df_train.iloc[:,1:].copy()
y = df_train.iloc[:,0].copy()
num_iter = 10000


def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def log_likelihood(x, y, weights):
    z = np.dot(x, weights)
    ll = np.sum( y*z - np.log(1 + np.exp(z)) )
    return ll

def gradient_ascent(X, h, y):
    return np.dot(X.T, y - h)

def update_weight_mle(weight, learning_rate, gradient):
    return weight + learning_rate * gradient

intercept2 = np.ones((X2.shape[0], 1))
X2 = np.concatenate((intercept2, X2), axis=1)
theta2 = np.zeros(X2.shape[1])

for i in range(num_iter):
    h2 = sigmoid(X2, theta2)
    gradient2 = gradient_ascent(X2, h2, y) #np.dot(X.T, (h - y)) / y.size
    theta2 = update_weight_mle(theta2, 0.1, gradient2)


result2 = sigmoid(df_test, theta2)

print("Test Accuracy:")
f2 = pd.DataFrame(result2)

print accuracy_score(df_test.iloc[:,0], f2)
from sklearn.linear_model import LogisticRegression as LReg
#lreg = LReg()
#lreg.fit(X, y)
# Use score method to get accuracy of model
#score = lreg.score(df_validation.iloc[:,1:], df_validation[0])
#print(score)

print "\n###### With regularization #######"

x_train = df_train.iloc[:,1:]
y_train = y = df_train.iloc[:,0]
x_validation = df_validation.iloc[:,1:]
y_validation = df_validation.iloc[:,0]
x_test = df_test.iloc[:,1:]
y_test = y = df_test.iloc[:,0]


C_l1 = [100, 10, 1, .1, .001]
print "With l1 penalty:"
for c in C_l1:
    clf_l1 = LReg(penalty='l1', C=c)
    clf_l1.fit(x_train, y_train)
    print('C:', c)
    #print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf_l1.score(x_train, y_train))
    print('Validation accuracy:', clf_l1.score(x_validation, y_validation))
    print "Weights:",clf_l1.coef_
    print('')
    
clf_l1 = LReg(penalty='l1', C=1)
clf_l1.fit(x_train, y_train)
print "Weights:",clf_l1.coef_
print('Test accuracy on c=1:', clf_l1.score(x_test, y_test)),"\n"

C_l2 = [100, 10, 1, .1, .001]
print "With l2 penalty:"
for c in C_l2:
    clf_l2 = LReg(penalty='l2', C=c)
    clf_l2.fit(x_train, y_train)
    print('C:', c)
    #print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy:', clf_l2.score(x_train, y_train))
    print('Validation accuracy:', clf_l2.score(x_validation, y_validation))
    print "Weights:",clf_l2.coef_
    print('')
    
clf_l2 = LReg(penalty='l2', C=1)
clf_l2.fit(x_train, y_train)
print "Weights:",clf_l2.coef_
print('Test accuracy on c=1:', clf_l2.score(x_test, y_test)),"\n"
