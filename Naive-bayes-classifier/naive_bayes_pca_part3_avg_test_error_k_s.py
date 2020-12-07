import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/sonar_train.csv", header=None)
df_validation = pd.read_csv("/Users/manasbundele/Downloads/sonar_valid.csv", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/sonar_test.csv", header=None)

data_wo_labels = df_train.iloc[:,0:60]
labels = df_train.iloc[:,60]


def separate_data_by_class(data):
    separated = {}
    for i in range(len(data)):
        row = data.iloc[i]
        label = row[60]
        if (label not in separated):
            separated[label] = []
        separated[label].append(row)
    return separated


def mean(data):
    return np.mean(data, axis=0)
 
def stdev(data):
    data_std = np.std(data, axis=0)
    return data_std

def summarize_data(data):
    summaries = []
    mean_arr = mean(data)
    stdev_arr = stdev(data)
    for i in range(len(mean_arr)):
        summaries.append((mean_arr[i], stdev_arr[i]))
    del summaries[-1]
    return summaries

def summarize_by_class(data):
    separated = separate_data_by_class(data)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_data(instances)
    return summaries

def calculate_probability(x, mean, stdev):
    #print x#,mean
    exponent = math.exp(-(float(math.pow(x-mean,2.0))/(2.0*math.pow(stdev,2.0))))
    return (1/(math.sqrt(2.0*math.pi)*stdev))*exponent

def calculate_class_probabilities(summaries, input_row):
    probabilities = {}
    keys = input_row.keys()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(input_row)-1):
            mean, stdev = class_summaries[keys[i]]
            x = input_row[keys[i]]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities
    
    
def predict(summaries, input_row):
    probabilities = calculate_class_probabilities(summaries, input_row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def prediction_for_dataset(summaries, test_data):
    predictions = []
    for i in range(len(test_data)):
        result = predict(summaries, test_data.iloc[i])
        predictions.append(result)
    return predictions

def accuracy(test_data, predictions):
    correct = 0
    for x in range(len(test_data)):
        last_key = test_data.keys()[-1]
        if test_data[last_key].iloc[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test_data)))

def compute_top_k_eig(data,k):
    data = df_train.iloc[:,0:60]
    data_std = StandardScaler().fit_transform(data)

    mean_vec = np.mean(data_std, axis=0)
    cov_mat = (data_std - mean_vec).T.dot((data_std - mean_vec)) / (data_std.shape[0]-1)
    
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    u,s,v = np.linalg.svd(data_std.T)
    
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()
    
    count = 0
    v=[]
    for i in eig_pairs:
        count = count + 1
        v.append(i[1]) 
        if count == k:
            break

    return v

def compute_pi(data,v,k):
    squared_v = np.square(v)
    pi=[]
    for j in range(len(data.iloc[0])-1):
        sum_squared_vj = 0
        for i in range(k):
            sum_squared_vj += squared_v[i][j]
        pi.append(float(sum_squared_vj)/k)
        
    return pi

def sample_features(data, pi_cdf, s):
    indices = []
    for i in range(s):
        index = random.uniform(0, 1)
        for j in range(len(pi_cdf)):
            if j<60 and index < pi_cdf[j+1] and index > pi_cdf[j]:
                if j not in indices:
                    indices.append(j)
                break
            elif index < pi_cdf[1]:
                indices.append(0)
                break
            elif index > pi_cdf[len(pi_cdf)-1]:
                indices.append(len(pi_cdf)-2)
                break
    indices.append(60)
    return data[indices]
    

def calculate_avg_test_error(train_data, test_data):
    error = []
    for k in range(10):
        v = compute_top_k_eig(train_data,k+1)
        pi = compute_pi(train_data,v,k+1)
        pi_cdf = [0]
    
        for i in range(len(train_data.iloc[0])-1):
            sum_temp = pi_cdf[i] + pi[i]
            pi_cdf.append(sum_temp)
        
        for s in range(20):
            err = 0
            for i in range(100):
                new_test_dataset = sample_features(test_data, pi_cdf, s+1)
                summaries = summarize_by_class(train_data)
                predictions = prediction_for_dataset(summaries, new_test_dataset)
                err += 1 - accuracy(new_test_dataset, predictions)
            error.append(err)
            print "k=",k+1,"s=",s+1
            print "Average error on test set is ",(err/100.0)

    print min(error)

#print separate_data_by_class(df_train)
#print mean(data_wo_labels)
#print stdev(data_wo_labels)
#print summarize_data(data_wo_labels)

#summaries = summarize_by_class(df_train)
#predictions = prediction_for_dataset(summaries, df_test)
#print "Accuracy on test set is ",accuracy(df_test, predictions)

#v = compute_top_k_eig(data_wo_labels,6)

#pi = compute_pi(data_wo_labels, v, 6)
calculate_avg_test_error(df_train, df_test) 