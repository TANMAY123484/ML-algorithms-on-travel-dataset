# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:41:24 2023

@author: Admin
"""

"""
Created on Thu Dec 2, 2021

Code Description: A python code to test the efficacy of stand-alone Decision Tree on the travel_choice dataset.


"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

'''
_______________________________________________________________________________


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
X_train, X_test, y_train, y_test_dt = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalization - Column-wise
X_train_norm = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) - np.min(X_train, 0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) - np.min(X_test, 0))
X_test_norm = X_test_norm.astype(float)

#Algorithm - Decision Tree
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-DT/RESULTS4/' 
    

MSL = np.load(RESULT_PATH+"/h_MSL.npy")[0]
MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
CCP = np.load(RESULT_PATH+"/h_CCP.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]


clf = DecisionTreeClassifier(min_samples_leaf = MSL, random_state = 42, max_depth = MD, ccp_alpha = CCP)
clf.fit(X_train_norm, y_train.ravel())


y_pred_dt = clf.predict(X_test_norm)
f1 = f1_score(y_test_dt, y_pred_dt, average='macro')

print('TRAINING F1 Score', F1SCORE)
print('TESTING F1 Score', f1)

np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )

##################################################################

# Fetching feature importances and their respective names
importances = clf.feature_importances_
feature_names = df1.columns[:-1]  # because the last column is 'Mode' which is our target

# Pairing feature names with their importances
feature_importances = sorted(zip(importances, feature_names), reverse=True)

# Printing the paired feature importances
for importance, name in feature_importances:
    print(f"Feature: {name}, Importance: {importance:.4f}")

# Optional: Plotting feature importances
plt.figure(figsize=(12, 8))
plt.barh([name for importance, name in feature_importances], 
         [importance for importance, name in feature_importances])
plt.xlabel("Decision tree Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()

####################################################

















