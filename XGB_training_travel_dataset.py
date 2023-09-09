# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:36:13 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:54:01 2023

@author: Admin
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# Read the travel dataset
df=pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles','HH_inc', 'av_metro','tt', 'tc','Mode']
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




#smote = SMOTE(random_state=42)
#X_train_balanced, y_train_balanced = smote.fit_resample(X_train_norm, y_train)
# Set the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [1, 3],
    'learning_rate': [0.1],
    'gamma': [0],
    'n_estimators': [1000],
    'min_child_weight': [1]
}

BEST_F1 = 0

for max_depth in param_grid['max_depth']:
    for learning_rate in param_grid['learning_rate']:
        for gamma in param_grid['gamma']:
            for n_estimators in param_grid['n_estimators']:
                for min_child_weight in param_grid['min_child_weight']:
                    FSCORE_TEMP = []

                    for TRAIN_INDEX, VAL_INDEX in KFold(n_splits=5, random_state=42, shuffle=True).split(X_train_norm):
                        X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
                        Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]

                        model = xgb.XGBClassifier(
                            objective='multi:softmax',
                            num_class=len(np.unique(y)),
                            eval_metric='merror',
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            gamma=gamma,
                            n_estimators=n_estimators,
                            min_child_weight=min_child_weight
                        )
                        model.fit(X_TRAIN, Y_TRAIN.ravel())
                        Y_PRED = model.predict(X_VAL)
                        f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                        FSCORE_TEMP.append(f1)
                        print('F1 Score:', f1)

                    mean_f1 = np.mean(FSCORE_TEMP)
                    print("Mean F1-Score for max_depth =", max_depth, "learning_rate =", learning_rate,
                          "gamma =", gamma, "n_estimators =", n_estimators,
                          "min_child_weight =", min_child_weight, "is", mean_f1)

                    if mean_f1 > BEST_F1:
                        BEST_F1 = mean_f1
                        BEST_PARAMS = {
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'gamma': gamma,
                            'n_estimators': n_estimators,
                            'min_child_weight': min_child_weight
                        }

print("Best F1 Score:", BEST_F1)
print("Best Hyperparameters:", BEST_PARAMS)

# Define the directory path to save the results
RESULTS_PATH = os.getcwd() + '/SA-XG-BOOST-NEW/RESULTS/'

# Check if the directory exists, if not, create it
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Save the best F1 score
np.save(os.path.join(RESULTS_PATH, 'best_f1_score.npy'), BEST_F1)

# Save the best hyperparameters
np.save(os.path.join(RESULTS_PATH, 'best_hyperparameters.npy'), BEST_PARAMS)

print("Hyperparameter tuning results saved successfully.")
print('BEST F1 Score during training:', BEST_F1)
print('Best Hyperparameters:')
for key, value in BEST_PARAMS.items():
    print(key + ':', value)