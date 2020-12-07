import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/mush_train.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/mush_test.data", header=None)

attribute_names_dict = ['class',
'cap-shape',
'cap-surface',
'cap-color',
'bruises',
'odor',
'gill-attachment',
'gill-spacing',
'gill-size',
'gill-color',
'stalk-shape', 
'stalk-root',
'stalk-surface-above-ring',
'stalk-surface-below-ring',
'stalk-color-above-ring',
'stalk-color-below-ring',
'veil-type',
'veil-color',
'ring-number',
'ring-type',
'spore-print-color',
'population',
'habitat']

df_train.columns = attribute_names_dict
df_test.columns = attribute_names_dict

def entropy(target):
    distinct_elements, counts = np.unique(target,return_counts = True)
    entropy = np.sum([(-float(counts[i])/np.sum(counts))*np.log2(float(counts[i])/np.sum(counts)) for i in range(len(distinct_elements))])
    return entropy

def info_gain(samples, split_attr):
    # 0th column is the target
    total_entropy = entropy(samples['class'])
    distinct_elements, counts = np.unique(samples[split_attr],return_counts=True)
    weighted_entropy = np.sum([float(counts[i])/np.sum(counts) * entropy(samples.where(samples[split_attr] == distinct_elements[i]).dropna()['class']) for i in range(len(distinct_elements))])
    
    # Calculate the information gain
    infogain = total_entropy - weighted_entropy

    return infogain

# features = df.columns[1:]
def ID3(data, samples, features):
    
    # if all the target values have same value
    if len(np.unique(data['class'])) <= 1:
        return np.unique(data['class'])[0]
    
    # if the dataset is empty, return the target feature value in the original dataset
    elif len(data)==0:
        return np.unique(samples['class'])[np.argmax(np.unique(samples['class'],return_counts=True)[1])]
    
    elif len(features) ==0:
        return None
    
    else:
        # select the feature that best splits the dataset
        ig_of_features = [info_gain(data, feature) for feature in features]
        
        indexes_of_max_ig = [i for i,val in enumerate(ig_of_features) if val == max(ig_of_features)]
        best_feature = features[indexes_of_max_ig[-1]]
        
        # Tree structure
        tree = {best_feature:{}}
        
        # remove the features witht he best information gain
        features = [i for i in features if i != best_feature]
        
        # grow a branch under the root node for every possible value of root node feature
        for value in np.unique(data[best_feature]):
            subdata = data.where(data[best_feature] == value).dropna()
            subtree = ID3(subdata, samples, features)
            tree[best_feature][value] = subtree
        return (tree)
        
def predict(query, tree):
    for key in list(query.keys()):
       if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            
            except:
                # for the case we cannot correctly classify, 
                return 'p'
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        
        
def test(test_data, tree):
    queries = test_data.iloc[:,1:].to_dict(orient = "records")
    prediction = pd.DataFrame(columns=["prediction"])
    for i in range(len(test_data)):
        prediction.loc[i,"prediction"] = predict(queries[i],tree) 
    print 'The prediction accuracy is:',(float(np.sum(prediction["prediction"] == test_data["class"]))/len(test_data))*100


if __name__ == "__main__":
    tree = ID3(df_train,df_train,df_train.columns[1:])
    print(tree)
    test(df_train, tree)

    # Part (7) below
    train_test_data = pd.concat([df_train, df_test])
    new_train, new_test, y, y_test = train_test_split(train_test_data, train_test_data['class'],test_size=0.1)
    test(new_train.reset_index(drop=True), tree)
    test(new_test.reset_index(drop=True),tree)