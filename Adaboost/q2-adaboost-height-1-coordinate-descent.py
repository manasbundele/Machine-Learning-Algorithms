import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import random

# Put data into dataframe
df_train = pd.read_csv("/Users/manasbundele/Downloads/heart_train.data", header=None)
df_test = pd.read_csv("/Users/manasbundele/Downloads/heart_test.data", header=None)

H=dict()

# 0N0
def apply_tree_1(data, atr, datapoint=None):
    h = []
    df = data
    if datapoint == None:    
        for tdata in range(len(data)):
            if (df[atr][tdata] == 0 or df[atr][tdata] == 1):
                h.append(0)
    
        return h,1
    else: 
        if (df[atr][datapoint] == 0 or df[atr][datapoint] == 1):
            return 0
    
# NLL
def apply_tree_2(data,atr,datapoint=None):
    h = []
    df = data
    if datapoint == None:     
        for tdata in range(len(data)):
            if (df[atr][tdata] == 0 or df[atr][tdata] == 1):
                h.append(1)
        
        return h,2
    else:
        if (df[atr][datapoint] == 0 or df[atr][datapoint] == 1):
                return 1

# NRL
def apply_tree_3(data, atr,datapoint=None):
    h = []
    df = data
    if datapoint == None:     
        for tdata in range(len(data)):
            if (df[atr][tdata] == 0):
                h.append(0)
            elif df[atr][tdata] == 1:
                h.append(1)
        return h,3
    else:
        if (df[atr][datapoint] == 0):
                return 0
        elif df[atr][datapoint] == 1:
                return 1
    
# NLR
def apply_tree_4(data, atr,datapoint=None):
    h = []
    df = data
    if datapoint == None:
        for tdata in range(len(data)):
            if (df[atr].loc[tdata] == 0):
                h.append(1)
            elif df[atr][tdata] == 1:
                h.append(0)
        
        return h,4
    else:
        if (df[atr][datapoint] == 0):
                return 1
        elif df[atr][datapoint] == 1:
                return 0




def fit(df_train):
    bin0 = 0
    bin1 = 1
    df_train_x = df_train.iloc[:,1:]
    df_train_y = df_train[0].values
    alpha = 0.2
    for t in range(5):
        Hypothesis_space = dict()
        #for i in range(len(df_train_x.columns)):
        atr = random.choice([i for i in [1,len(df_train_x.columns)]])
        r = random.choice([1,2,3,4])
        print 'r=',r, 'atr=',atr
        if r == 1:
            ht_,t_ = apply_tree_1(df_train_x, atr)
        elif r==2:
            ht_,t_ = apply_tree_2(df_train_x, atr)
        elif r==3:
            ht_,t_ = apply_tree_3(df_train_x, atr)
        elif r==4:
            ht_,t_ = apply_tree_4(df_train_x, atr)
            
        misclassified_ht_ = np.where(ht_ != df_train_y,1,0)
        num =0.0
        den =0.0
        for j in range(len(df_train_x)):
            sum_rest=0
            for i in range(len(df_train_x.columns)):
                atr_temp = i + 1
                if atr_temp != atr and r != 1: 
                    sum_rest += alpha * apply_tree_1(df_train_x, atr_temp, df_train_x[i+1][j])
                if atr_temp != atr and r != 2:
                    sum_rest += alpha * apply_tree_2(df_train_x, atr_temp, df_train_x[i+1][j])
                if atr_temp != atr and r != 3:
                    sum_rest += alpha * apply_tree_3(df_train_x, atr_temp, df_train_x[i+1][j])
                if atr_temp != atr and r != 4:
                    sum_rest += alpha * apply_tree_4(df_train_x, atr_temp, df_train_x[i+1][j])
                
            if misclassified_ht_[j] == 0:
                num += np.exp(-1 * df_train_y[j] * sum_rest)
                #print 'num=',num
            else:
                den += np.exp(-1 * df_train_y[j] * sum_rest)
                #print 'den=',den

        alpha_t = 0.5 * np.log(float(num)/den)
        print 'alpha_t=',alpha_t
        alpha = alpha_t
        H[alpha] = [-1 if x==0 else x for x in ht_]
        #print H[alpha]
        
    sum1=np.array([0.0 for i in range(len(df_train))])
    for key,lis in H.items():
        sum1 += key * np.array(lis)
        #print sum1

    h = np.sign(sum1)
    print 'h=',h
      
fit(df_train) 
