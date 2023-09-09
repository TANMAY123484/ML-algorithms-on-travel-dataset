# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:37:34 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:21:58 2023

@author: Admin
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import pdpbox
from pdpbox import pdp


from sklearn.metrics import confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import plot_partial_dependence

#from pdpbox import pdp, info_plots
#from imblearn.over_sampling import SMOTE

#Read the dataset
df=pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles','HH_inc','av_metro', 'tt', 'tc','Mode']
df1 = df[columns]

# Read the travel dataset
X = np.array(df1.iloc[:, 0:10])
y = np.array(df1.iloc[:, -1])
y = y.reshape(-1, 1)
y=y-1

# Split the dataset for training and testing (70-30)
X_train, X_test, y_train, y_test_xgb = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalization - Column-wise
X_train_norm = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) - np.min(X_train, 0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = (X_test - np.min(X_test, 0)) / (np.max(X_test, 0) - np.min(X_test, 0))
X_test_norm = X_test_norm.astype(float)


#smote = SMOTE(random_state=42)
#X_train_balanced, y_train_balanced = smote.fit_resample(X_train_norm, y_train)

# Algorithm - XGBoost
RESULT_PATH = os.getcwd() + '/SA-XG-BOOST-NEW/RESULTS/'

BEST_PARAMS = {
    'max_depth': np.load(os.path.join(RESULT_PATH, 'best_hyperparameters.npy'), allow_pickle=True).item().get('max_depth'),
    'learning_rate': np.load(os.path.join(RESULT_PATH, 'best_hyperparameters.npy'), allow_pickle=True).item().get('learning_rate'),
    'gamma': np.load(os.path.join(RESULT_PATH, 'best_hyperparameters.npy'), allow_pickle=True).item().get('gamma'),
    'n_estimators': np.load(os.path.join(RESULT_PATH, 'best_hyperparameters.npy'), allow_pickle=True).item().get('n_estimators'),
    'min_child_weight': np.load(os.path.join(RESULT_PATH, 'best_hyperparameters.npy'), allow_pickle=True).item().get('min_child_weight')
}


model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eval_metric='merror',
    **BEST_PARAMS
)

model.fit(X_train_norm, y_train.ravel())

y_pred_xgb = model.predict(X_test_norm)

# Load the best F1 score obtained during training
BEST_F1 = np.load(os.path.join(RESULT_PATH, 'best_f1_score.npy'))

# Compute the F1 score on the testing data
f1 = f1_score(y_test_xgb, y_pred_xgb, average='macro')

print('TESTING F1 Score:', f1)
print('BEST F1 Score during training:', BEST_F1)



RESULT_PATH2 = os.getcwd() + '/SA-XG-BOOST-NEW/RESULTS'
BEST_F1 = np.load(os.path.join(RESULT_PATH2, 'best_f1_score.npy'))


#####################################################################
# Feature Importance
sorted_idx = np.argsort(model.feature_importances_)
feature_names = ['av_metro', 'Female', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles', 'tt', 'tc', 'HH_inc']

#feature_names = df1.columns[:-1]



plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
plt.barh(np.array(feature_names)[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Feature Importance")
plt.show()




