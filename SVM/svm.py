import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers

# Put data into dataframe
df = pd.read_csv("/Users/manasbundele/Downloads/mystery.data")

# Extract features and labels into different matrices
feature_matrix = df.iloc[:,0:4].values
label_matrix = df.iloc[:,4].values

def fit(features, labels):
    num_data, num_features = features.shape
    
    # Parameters of QP prob
    K = np.dot(features, features.transpose())
    P = cvxopt.matrix(np.outer(labels,labels) * K)
    q = cvxopt.matrix(-1 * np.ones(num_data))
    A = cvxopt.matrix(labels, (1, num_data), 'd')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(-1 * np.ones(num_data)))
    h = cvxopt.matrix(np.zeros(num_data))
    
    # Solving QP Prob
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
    # Lagrange multipliers
    a = np.ravel(solution['x'])
    
    support_vec = a > 1e-5
    ind = np.arange(len(a))[support_vec]
    a = a[support_vec]
    print 'Support Vectors = ', a
    print 'Number of support vectors = ', len(list(a))
    support_vec_features = features[support_vec]
    support_vec_labels = labels[support_vec]
    
    # Intercept
    b = 0
    for n in range(len(a)):
        b += support_vec_labels[n]
        b -= np.sum(a * support_vec_labels * K[ind[n], support_vec])
    b /= len(a)

    # Weight vector
    w = np.zeros(num_features)
    for n in range(len(a)):
        w += a[n] * support_vec_labels[n] * support_vec_features[n]

    return w, b


if __name__ == "__main__":
    # Calculating weights and bias
    w, b = fit(feature_matrix, label_matrix)
    print 'weight = ', w, '\nbias = ', b

    norm_w = np.linalg.norm(w)
    print 'norm_w =', norm_w
        
    max_margin = 1/norm_w
    print 'Max margin = ',max_margin
        
        

