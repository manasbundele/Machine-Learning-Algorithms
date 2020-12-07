import numpy as np
import pandas as pd
import cvxopt
import cvxopt.solvers

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/park_train.data", header=None)
df_validation = pd.read_csv("/Users/manasbundele/Downloads/park_validation.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/park_test.data", header=None)

# Extract features and labels into different matrices
feature_matrix = df_train.iloc[:,1:23].values
label_matrix = df_train.iloc[:,0].values
feature_matrix_validation = df_validation.iloc[:,1:23].values
label_matrix_validation = df_validation.iloc[:,0].values
feature_matrix_test = df_test.iloc[:,1:23].values
label_matrix_test = df_test.iloc[:,0].values

def fit(features, labels, C):
    num_data, num_features = features.shape
    
    # Parameters of QP prob
    K = np.dot(features, features.transpose())
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
    print 'Number of support vectors = ', len(list(support_vec_features))
    
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
    return normalize_data(w), b


def predict(data, w, b):
    return np.sign(np.dot(data, w) + b)
    
def normalize_data(data): 
    data_norm = (data - data.mean()) / (data.max() - data.min())
    data_norm[0]= data[0]
    return data_norm

def test(x_train, y_train, x_test, y_test, C):
    w,b= fit(x_train, y_train, C)
    y_predict = predict(x_test,w,b)
    correct = np.sum(y_predict == y_test)
    #print("%d out of %d predictions correct for C = %d" % (correct, len(y_predict), C))
    print 'Accuracy = %f percent for C=%d' %(100*(float(correct)/ len(y_predict)), C)

if __name__ == "__main__":
    
    C = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]

    print '*********Training dataset*****************'
    for i in range(len(C)):
        test(feature_matrix, label_matrix, feature_matrix, label_matrix, C[i])

    print '*********Validation dataset*****************'
    for i in range(len(C)):
        test(feature_matrix, label_matrix, feature_matrix_validation, label_matrix_validation, C[i])

    print '*********Test dataset*****************'
    test(feature_matrix, label_matrix, feature_matrix_test, label_matrix_test, 1000)
