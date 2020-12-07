import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/heart_train.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/heart_test.data", header=None)

print df_train

Hypothesis_space = dict()
Weights = [1/float((len(df_train))) for i in range(len(df_train))]
print Weights
H = dict()

# N denote the root-node, L denote the left node and R denote the right node. 
# NRR
def apply_tree_1(data, atr1, atr2, atr3):
    h = []
    df = data
    for tdata in range(len(data)):
        if (df[atr1][tdata] == 0 or (df[atr1][tdata] == 1 and df[atr2][tdata] == 0) or (df[atr1][tdata] == 1 and df[atr2][tdata] == 1 and df[atr3][tdata] == 0)):
            h.append(0)
        elif (df[atr1][tdata] == 1 and df[atr2][tdata] == 1 and df[atr3][tdata] == 1):
            h.append(1) 

    return h,1
    
# NLL
def apply_tree_2(data,atr1, atr2, atr3):
    h = []
    df = data
    for tdata in range(len(data)):
        if df[atr1][tdata] == 1 or (df[atr1][tdata] == 0 and df[atr2][tdata] == 1) or (df[atr1][tdata] == 0 and df[atr2][tdata] == 0 and df[atr3][tdata] == 1):
            h.append(1)
        elif (df[atr1][tdata] == 0 and df[atr2][tdata] == 0 and df[atr3][tdata] == 0):
            h.append(0)
    
    return h,2

# NRL
def apply_tree_3(data, atr1, atr2, atr3):
    h = []
    df = data
    for tdata in range(len(data)):
        if df[atr1][tdata] == 0 or (df[atr1][tdata] == 1 and df[atr2][tdata] == 0 and df[atr3][tdata] == 0):
            h.append(0)
        elif (df[atr1][tdata] == 1 and df[atr2][tdata] == 1) or (df[atr1][tdata] == 1 and df[atr2][tdata] == 0 and df[atr3][tdata] == 1):
            h.append(1)
    
    return h,3
    
# NLR
def apply_tree_4(data, atr1, atr2, atr3):
    h = []
    df = data
    for tdata in range(len(data)):
        if (df[atr1][tdata] == 0 and df[atr2][tdata] == 0) or (df[atr1][tdata] == 0 and df[atr2][tdata] == 1 and df[atr3][tdata] == 0):
            h.append(0)
        elif df[atr1][tdata] == 1 or (df[atr1][tdata] == 0 and df[atr2][tdata] == 1 and df[atr3][tdata] == 1):
            h.append(1)
    
    return h,4
    
    
# LNR
def apply_tree_5(data, atr1, atr2, atr3):
    h = []
    df = data
    for tdata in range(len(data)):
        if (df[atr1][tdata] == 0 and df[atr2][tdata] == 0) or (df[atr1][tdata] == 1 and df[atr3][tdata] == 0):
            h.append(0)
        elif (df[atr1][tdata] == 0 and df[atr2][tdata] == 1) or (df[atr1][tdata] == 1 and df[atr3][tdata] == 1):
            h.append(1)

    return h,5




def fit(df_train,number_of_rounds):
    bin0 = 0
    bin1 = 1
    df_train_x = df_train.iloc[:,1:]
    df_train_y = df_train[0].values
    
    for t in range(number_of_rounds):
        for i in range(len(df_train_x.columns)):
            for j in range(len(df_train_x.columns)):
                for k in range(len(df_train_x.columns)):
                    [atr1, atr2, atr3] = df_train_x.columns.to_series().sample(3)
                    h1, t1 = apply_tree_1(df_train_x, atr1, atr2, atr3)
                    err1 = np.sum([Weights[m] for m in range(len(df_train_y)) if h1[m] != df_train_y[m]])/float(len(df_train))
                    
                    h2, t2 = apply_tree_2(df_train_x, atr1, atr2, atr3)
                    err2 = np.sum([Weights[m] for m in range(len(df_train_y)) if h2[m] != df_train_y[m]])/float(len(df_train))
                    
                    h3, t3 = apply_tree_3(df_train_x, atr1, atr2, atr3)
                    err3 = np.sum([Weights[m] for m in range(len(df_train_y)) if h3[m] != df_train_y[m]])/float(len(df_train))
                    
                    h4, t4 = apply_tree_4(df_train_x, atr1, atr2, atr3)
                    err4 = np.sum([Weights[m] for m in range(len(df_train_y)) if h4[m] != df_train_y[m]])/float(len(df_train))
                    
                    h5, t5 = apply_tree_5(df_train_x, atr1, atr2, atr3)
                    err5 = np.sum([Weights[m] for m in range(len(df_train_y)) if h5[m] != df_train_y[m]])/float(len(df_train))
                    
                    Hypothesis_space[err1] = [h1, t1, [atr1,atr2,atr3]]
                    Hypothesis_space[err2] = [h2, t2, [atr1,atr2,atr3]]
                    Hypothesis_space[err3] = [h3, t3, [atr1,atr2,atr3]]
                    Hypothesis_space[err4] = [h4, t4, [atr1,atr2,atr3]]
                    Hypothesis_space[err5] = [h5, t5, [atr1,atr2,atr3]]
                    
        min_err = min(Hypothesis_space.keys())
        min_err_hypothesis = Hypothesis_space[min_err] 
        print min_err_hypothesis
        print 'err=',min_err
        alpha = np.log((1-min_err)/float(min_err))
        #print alpha
        print 'alpha=', alpha
        misclassified = np.where(min_err_hypothesis[0] != df_train_y,1,0)
        for i in range(len(Weights)):
            Weights[i] *= np.exp(alpha * misclassified[i])/2* np.sqrt(float(min_err) * (1-min_err))
        
        H[alpha] = [min_err_hypothesis[0], min_err]
        Hypothesis_space.pop(min_err)

    print 'Done'    
    return H


H=fit(df_train,5)

H=fit(df_train,10) 
#predict(df_train,H)
#predict(df_test,H)
