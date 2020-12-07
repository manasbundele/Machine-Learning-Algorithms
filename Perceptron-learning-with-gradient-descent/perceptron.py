import random
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/manasbundele/Downloads/perceptron.data')

feature_mat = df.iloc[:,0:4].values
label_mat = df.iloc[:,4].values

# learning rate: 1, initial weights: 0
# Standard Gradient Descent
def stdgd_perceptron(features, labels):
    num_examples, num_features = features.shape
    # storing w,b for each iteration
    dummy_w = [[0.0,0.0,0.0,0.0] for i in range(999)]
    dummy_b = [[0.0] for i in range(999)]
    w = [0.0, 0.0, 0.0, 0.0]
    b = 0
    converged = False
    b = 0
    it = 0
    max_iter = 999
    while (it < max_iter):
        w_updated = False
        del_w = []
        del_b = []
        
        for i in range(0, num_examples):
            a = b + np.dot(w, features[i])
            if np.sign(labels[i] * a) != 1:
                del_w.append(features[i] * labels[i])
                del_b.append(labels[i])
            else:
                del_w.append([0.0,0.0,0.0,0.0])
                del_b.append(0)
        
        dummy_w[it] += np.sum(del_w, axis=0)
        dummy_b[it] += np.sum(del_b)
        w += dummy_w[it]
        b += dummy_b[it]
        
        if set(dummy_w[it]) != 0.0:
            w_updated = True

        if not w_updated:
           print("Convergence reached in %i iterations." % it)
           converged = True
           break
        
        if it == 1:
            print 'w1 = ', w, '\nb1 = ', b
        elif it == 2:
            print 'w2 = ', w, '\nb2 = ', b
        elif it == 3:
            print 'w3 = ', w, '\nb3 = ', b
        
        it += 1
    return w, b

# learning rate: 1, initial weights: 0
# Stochastic Gradient Descent - Perceptron
def sgd_perceptron(features, labels):
    num_examples, num_features = features.shape
    w = np.zeros(num_features)
    converged = False
    b = 0
    it = 0
    while not converged:
        w_updated = False
        for i in range(0, num_examples):
            a = b + np.dot(w, features[i])
            if np.sign(labels[i] * a) != 1:
                w_updated = True
                w += features[i] * labels[i]
                b += labels[i]
        if not w_updated:
            print("Converged in %i iterations." % it)
            converged = True
            break
        if it == 1:
            print 'w1 = ', w, '\nb1 = ', b
        elif it == 2:
            print 'w2 = ', w, '\nb2 = ', b
        elif it == 3:
            print 'w3 = ', w, '\nb3 = ', b
        
        it += 1
    return w, b


if __name__ == "__main__":

    final_w, final_b = sgd_perceptron(feature_mat, label_mat)
    print '\nStochastic GD: ', 'w = ', final_w, '\nb = ', final_b , "\n\n\n"



    final_w, final_b =  stdgd_perceptron(feature_mat, label_mat)    
    print '\nStandard GD: ', 'w = ', final_w, '\nb = ', final_b
