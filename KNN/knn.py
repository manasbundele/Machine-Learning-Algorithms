import numpy as np
import pandas as pd
import math
import operator

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/park_train.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/park_test.data", header=None)
df_validation = pd.read_csv("/Users/manasbundele/Downloads/park_validation.data", header=None)

def euclidean_distance(inst1, inst2):
    distance = 0
    for i in range(1,len(df_train.columns)):
        distance += np.square(inst1[i] - inst2[i])
    return np.sqrt(distance)

def get_neighbors(train_dataset_points, test_feature_point, k):
    distances_arr = []
    length = len(test_feature_point) - 1
    for i in range(len(train_dataset_points)):
        dist = euclidean_distance(test_feature_point, train_dataset_points.iloc[i])
        distances_arr.append((train_dataset_points.iloc[i], dist, i))
    
    distances_arr.sort(key=operator.itemgetter(1))
    neighbors_arr = []
    for i in range(k):
        neighbors_arr.append(distances_arr[i][2])
         
    return neighbors_arr

def predict(train_dataset_points, test_feature_point, k):
    
    k_nearest_neighbor_indexes = get_neighbors(train_dataset_points, test_feature_point, k)
    vote_count = {}
    for i in range(k):
        label = train_dataset_points.iloc[k_nearest_neighbor_indexes[i]][0]
        
        if label in vote_count:
            vote_count[label] += 1
        else:
            vote_count[label] = 1

    sorted_votes = sorted(vote_count.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_votes[0][0], k_nearest_neighbor_indexes)

def accuracy(train_dataset_points, test_feature_points, k):
    count_positive_res = 0
    total_feature_points = len(test_feature_points)
    for i in range(total_feature_points):
        result, neighbors = predict(train_dataset_points, test_feature_points.iloc[i], k)
        if (test_feature_points.iloc[i][0] == result):
            count_positive_res += 1
            
    accuracy = float(count_positive_res)/total_feature_points
    return accuracy
    
def normalize_data(data): 
    data_norm = (data - data.mean()) / (data.max() - data.min())
    data_norm[0]= data[0]
    return data_norm

if __name__ == "__main__":        
    # Setting number of neighbors
    k = 1
    print 'Accuracy for k=', k , 'is', accuracy(normalize_data(df_train), normalize_data(df_test), k)