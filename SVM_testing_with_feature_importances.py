# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:47:44 2023

@author: Admin
"""

import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance



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
X_train, X_test, y_train, y_test_svm = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalization - Column-wise
X_train_norm = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) - np.min(X_train, 0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) - np.min(X_test, 0))
X_test_norm = X_test_norm.astype(float)
#Algorithm - SVM
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-SVM/RESULTS_FINAL/' 
    

c = np.load(RESULT_PATH+"/h_C.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]



clf = SVC(C = c, kernel='rbf', decision_function_shape='ovr', random_state = 42)
clf.fit(X_train_norm, y_train.ravel())


y_pred_svm = clf.predict(X_test_norm)
f1 = f1_score(y_test_svm, y_pred_svm, average='macro')

print('TRAINING F1 Score', F1SCORE)
print('TESTING F1 Score', f1)



# Using your trained clf
result = permutation_importance(clf, X_test_norm, y_test_svm.ravel(), scoring='f1_macro', n_repeats=30, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 7))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(columns[:-1])[sorted_idx])
plt.title("Permutation Importances (SVM)")
plt.tight_layout()
plt.show()

# Extracting the mean importance values
importances = result['importances_mean']

# Sorting the importances
sorted_importances = importances[sorted_idx]

# Matching sorted importances with their corresponding column names
sorted_columns = np.array(columns)[sorted_idx]

# Plotting the results
plt.figure(figsize=(10, 7))
plt.barh(sorted_columns, sorted_importances, align='center')
plt.xlabel('SVM Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Plot')
plt.show()