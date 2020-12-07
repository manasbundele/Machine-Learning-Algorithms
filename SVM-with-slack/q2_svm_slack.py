
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


def gaussian_kernel(x, y, sigma=0.1):
    return np.exp(-float(np.linalg.norm(x-y)**2)/ (2 * (sigma ** 2)))

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

def test_gaussian(x_train, y_train, x_test, y_test, C, sigma):
    a, support_vec_features, support_vec_labels, b = fit(x_train, y_train, C, sigma)
    y_predict = predict(x_test,a, support_vec_features, support_vec_labels, b, sigma)
    correct = np.sum(y_predict == y_test)
    #print("%d out of %d predictions correct for C=%d and sigma=%f" % (correct, len(y_predict), C, sigma ))
    print 'Accuracy = %f percent for C=%d and sigma=%f' %(100*(float(correct)/ len(y_predict)), C, sigma)

if __name__ == "__main__":
    C = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
    sig = [0.1, 1, 10, 100, 1000]
    print '*********Training dataset*****************'
    for i in range(len(C)):
        for j in range(len(sig)):
            test_gaussian(feature_matrix, label_matrix, feature_matrix, label_matrix, C[i], sig[j])

    print '*********Validation dataset*****************'
    for i in range(len(C)):
        for j in range(len(sig)):
            test_gaussian(feature_matrix, label_matrix, feature_matrix_validation, label_matrix_validation, C[i], sig[j])

    print '*********Test dataset*****************'
    test_gaussian(feature_matrix, label_matrix, feature_matrix_test, label_matrix_test, 1000, 1000)
