# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:40:41 2023

@author: Admin
"""

"""
Created on Thu Dec 2, 2021


"""
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________
    Metro -1
    Bus   -2
    SR    -3
    Auto  -4
    Bike  -5
    Car   -6
    Cycle -7
    Walk  -8
    
    
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    df            -   Complete travel mode dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).

_______________________________________________________________________________

ML hyperparameter description:
_______________________________________________________________________________
    MSL     -   The minimum number of samples required to be at a leaf node.
    MD      -   The maximum depth of the tree.
    CCP     -   Complexity parameter used for Minimal Cost-Complexity Pruning. 
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________
    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

'''



# Read the travel dataset



df=pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles','HH_inc', 'tt', 'tc','av_metro','Mode']
df1 = df[columns]

# Split the dataset into features (X) and target (y) arrays
X = np.array(df1.iloc[:, 0:10])
y = np.array(df1.iloc[:, -1])
y = y.reshape(-1, 1)
y=y-1

# Split the dataset for training and testing (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalization - Column-wise
X_train_norm = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) - np.min(X_train, 0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) - np.min(X_test, 0))
X_test_norm = X_test_norm.astype(float)



#Algorithm - Decision Tree
clf = DecisionTreeClassifier(random_state=42)
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
alpha = ccp_path['ccp_alphas']
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train_norm) 
print(KF) 
for MSL in range(1,20):
                
    for MD in range(1,20):
        
        for CCP in alpha:

            FSCORE_TEMP=[]
        
            for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
                
                X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
                Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
            
                
                clf = DecisionTreeClassifier(min_samples_leaf = MSL, random_state=42, max_depth = MD, ccp_alpha = CCP)
                clf.fit(X_TRAIN, Y_TRAIN.ravel())
                Y_PRED = clf.predict(X_VAL)
                f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                FSCORE_TEMP.append(f1)
                print('F1 Score', f1)
            print("Mean F1-Score for MSL = ", MSL," MD = ", MD," CCP = ", CCP," is  = ",  np.mean(FSCORE_TEMP)  )
            if(np.mean(FSCORE_TEMP) > BESTF1):
                BESTF1 = np.mean(FSCORE_TEMP)
                BESTMSL = MSL
                BESTMD = MD
                BESTCCP = CCP
        
print("BEST F1SCORE", BESTF1)
print("BEST MD = ", BESTMD)
print("BEST MSL = ", BESTMSL)
print("BEST CCP = ", BESTCCP)


print("Saving Hyperparameter Tuning Results") 
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-DT/RESULTS4/'



try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_MSL.npy", np.array([BESTMSL]) ) 
np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]) ) 
np.save(RESULT_PATH+"/h_CCP.npy", np.array([BESTCCP]) ) 
np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 
