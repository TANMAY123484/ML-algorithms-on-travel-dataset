# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:30:26 2023

@author: Admin
"""





'''
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
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


#import the Dataset 
df=pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles','HH_inc', 'tt', 'tc','av_metro','Mode']
df1 = df[columns]


# Split the dataset into features (X) and target (y) arrays
X = np.array(df1.iloc[:, 0:10])
y = np.array(df1.iloc[:, -1])
y = y.reshape(-1, 1)
y=y-1


#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test_rf = train_test_split(X,y,test_size=0.3, random_state=42)

type(X_train)

#Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0)) / (np.max(X_train,0) - np.min(X_train,0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test,0)) / (np.max(X_test,0) - np.min(X_test,0))
X_test_norm = X_test_norm.astype(float)


# Apply SMOTE to balance the classes
#smote = SMOTE(random_state=42)
#X_train_balanced, y_train_balanced = smote.fit_resample(X_train_norm, y_train)


#Algorithm - Random Forest
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-RandomForest/RESULTS/' 
    

NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = RandomForestClassifier( n_estimators = 100, max_depth = 10, random_state=42)
clf.fit(X_train_norm, y_train.ravel())


y_pred_rf = clf.predict(X_test_norm)
f1 = f1_score(y_test_rf, y_pred_rf, average='macro')

print('TRAINING F1 Score ', F1SCORE)
print('TESTING F1 Score', f1)

###Feature importance
np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
import matplotlib.pyplot as plt
sorted_idx = clf.feature_importances_.argsort()
sorted_idx
sort_desc=np.argsort(-(sorted_idx))
plt.figure(figsize=(10,10))
plt.barh(df1.columns[sorted_idx], clf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show

