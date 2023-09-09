import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def plot_ice(tc_values, ice_values, labels, xlabel, title):
    plt.figure(figsize=(20, 15))
    for idx, mode in enumerate(mode_indices):
        plt.subplot(2, 2, idx + 1)
        
        # Plot the original curve and the modified curves
        for i, (label, values) in enumerate(zip(labels, ice_values)):
            plt.plot(tc_values, values[mode, :], label=label, linewidth=2)

        plt.title(title[idx], fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Probability', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=11, fontweight='bold')
        plt.yticks(fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Read the dataset
df = pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles', 'HH_inc', 'av_metro', 'tt', 'tc', 'Mode']
df1 = df[columns]

# Data Preparation
X = df1.drop(columns=['Mode'])
y = df1['Mode']

# Remove target classes 7 and 8
valid_targets = [1, 2, 3, 4, 5, 6]
X = X[y.isin(valid_targets)]
y = y[y.isin(valid_targets)]
y = y - 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

variables = [('tc', 'Travel Cost (in Rs)'), ('HH_inc', 'Household Income'), ('tt', 'Travel Time (in min)')]
modifications = [(1, 1.1, 1.2), (1, 1.1), (1, 0.9, 0.8)]

target_names = ['Metro', 'Bus', 'Two Wheeler', 'Car']
mode_indices = [0, 1, 4, 5]

for var, mod in zip(variables, modifications):
    feature_name, xlabel = var
    feature_min = X_train[feature_name].min()
    if feature_name == 'tt':
        feature_max = 100
    else:
        feature_max = X_train[feature_name].max()
    feature_values = np.linspace(feature_min, feature_max, 50)

    ice_values = [np.zeros((len(clf.classes_), len(feature_values))) for _ in mod]

    for i, target_class in enumerate(clf.classes_):
        for j, feature_value in enumerate(feature_values):
            test_data_copy = X_test.copy()
            for k, factor in enumerate(mod):
                test_data_copy[feature_name] = feature_value * factor
                prob_class = clf.predict_proba(test_data_copy)[:, i]
                ice_values[k][i, j] = prob_class.mean()

    labels = ['Original'] + [f'+{int((m-1)*100)}%' if m > 1 else f'{int((1-m)*100)}% Decreased' for m in mod[1:]]
    plot_ice(feature_values, ice_values, labels, xlabel, target_names)

    # Calculate the average increase and decrease in probability
    avg_changes = []
    for k, factor in enumerate(mod):
        if factor != 1:
            change = ice_values[k] - ice_values[0]  # Difference from original
            avg_change_for_mode = change.mean(axis=1)  # Average difference for each mode
            avg_changes.append(avg_change_for_mode)

            # Print the results
            change_type = 'increase' if factor > 1 else 'decrease'
            print(f"Average {change_type} in predicted probabilities for {feature_name}:")
            for mode_name, avg_val in zip(target_names, avg_change_for_mode):
                print(f"  {mode_name}: {avg_val:.4f}")
            print("\n")