#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: manasbundele
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cvxopt
import cvxopt.solvers

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/sonar_train.csv", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/sonar_test.csv", header=None)
df_validation = pd.read_csv("/Users/manasbundele/Downloads/sonar_valid.csv", header=None)

data = df_train.iloc[:,0:60]
labels = df_train.iloc[:,60]

#print data, labels

def compute_top_k_eig(data,k):
    data_std = StandardScaler().fit_transform(data)
    data_std = data*10
    mean_vec = np.mean(data_std, axis=0)
    #cov_mat = (data_std - mean_vec).T.dot((data_std - mean_vec)) / (data_std.shape[0]-1)
    
    cov_mat = np.cov((data_std - mean_vec).T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], i) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
    
    #print('Eigenvalues in descending order:')
    count = 0
    v=[]
    for i in eig_pairs:
        count = count + 1
        v.append(i[1]) 
        #print i[0]
        if count == k:
            break

    return v

def fit(features, labels, C, sigma):
    num_data, num_features = features.shape
    
    # Parameters of QP prob
    # Gram matrix
    K = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            K[i,j] = gaussian_kernel(features[i], features[j], sigma)
    P = cvxopt.matrix(np.outer(labels,labels) * K)
    q = cvxopt.matrix(-1 * np.ones(num_data))
    A = cvxopt.matrix(labels, (1, num_data), 'd')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.vstack((np.diag(np.ones(num_data) * -1), np.identity(num_data))))
    h = cvxopt.matrix(np.hstack((np.zeros(num_data), np.ones(num_data) * C)))

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 200
    cvxopt.solvers.options['feastol']=1e-2
    
    # Solving QP Prob
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
    # Lagrange multipliers
    a = np.ravel(solution['x'])
    support_vec = a > 0
    ind = np.arange(len(a))[support_vec]
    a = a[support_vec]
    support_vec_features = features[support_vec]
    support_vec_labels = labels[support_vec]
    #print 'Support Vectors = ', support_vec_features
    #print 'Number of support vectors = ', len(list(support_vec_features))
    
    # Intercept
    b = 0
    for n in range(len(a)):
        b += support_vec_labels[n]
        b -= np.sum(a * support_vec_labels * K[ind[n], support_vec])
    b /= len(a)

    return a, support_vec_features, support_vec_labels, b
    

def predict(data, a, support_vec_features, support_vec_labels, b, sigma):
    y_prediction = np.zeros(len(data))
    for i in range(len(data)):
        s = 0
        for j in range(len(a)):
            s += a[j] * support_vec_labels[j] * gaussian_kernel(data[i], support_vec_features[j], sigma)
        y_prediction[i] = s
    return np.sign(y_prediction + b)


def perform_svm_slack_on_dataset(data, test_data):
    X_train = data.iloc[:,0:60]
    y_train = df_train.iloc[:,60]
    X_test = test_data.iloc[:,0:60]
    y_test = test_data.iloc[:,60]
    
    c=[1,10,100,1000]
    for k in range(6):
        eigvec = compute_top_k_eig(data,k+1)
        for j in range(len(c)):
            X_train = np.dot(data.iloc[:,0:60],np.transpose(eigvec))
            X_test = np.dot(test_data.iloc[:,0:60],np.transpose(eigvec))
            svm = SVC(kernel='linear', C=c[j], random_state=0)
            svm.fit(X_train, y_train)
                    
            y_pred = svm.predict(X_test)
            print "k=",k+1,"c=",c[j],"error=",1-accuracy_score(y_test, y_pred)
    

    
#print u


#print('Covariance matrix \n%s' %cov_mat)
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)
#print data_std
    
#compute_top_k_eig(data,6)    
print "error on validation set"
perform_svm_slack_on_dataset(data, df_validation)
print "\nerror on test set"
perform_svm_slack_on_dataset(data, df_test)