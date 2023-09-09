# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:38:42 2023

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Your dataset loading code
df = pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles', 'HH_inc', 'av_metro', 'tt', 'tc', 'Mode']
df1 = df[columns]
df_filtered = df1[df1['Mode'].isin([2, 5, 8])]

X = df_filtered.drop(columns=['Mode'])
y = df_filtered['Mode']

# Apply t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)

# Set the plot style (optional: you can experiment with other styles)
plt.style.use('ggplot')

# Increase the figure size
plt.figure(figsize=(12, 8))

# Scatter plots with larger, semi-transparent markers
plt.scatter(X_tsne[y == 2, 0], X_tsne[y == 2, 1], label='Bus', color='red', s=50, alpha=0.6)
plt.scatter(X_tsne[y == 5, 0], X_tsne[y == 5, 1], label='2w', color='green', s=50, alpha=0.6)
plt.scatter(X_tsne[y == 8, 0], X_tsne[y == 8, 1], label='walk', color='blue', s=50, alpha=0.6)

# Adding x and y axis labels
plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)

# Adding a title
plt.title('t-SNE Visualization of Modes ', fontsize=16, fontweight='bold')

# Adding a legend
plt.legend(fontsize=12)

# Displaying the grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.show()

# Train a DecisionTreeClassifier on the t-SNE transformed data
clf = DecisionTreeClassifier()
clf.fit(X_tsne, y)

# Plotting the decision boundaries
x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
palette = sns.color_palette("husl", 3)
sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], hue=y, palette=palette, edgecolor="k", alpha=0.7)
plt.title('t-SNE visualization with decision boundaries')
plt.show()

###########3d plot####################
# Import the library
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
# Your dataset loading code
df = pd.read_csv('dataset1.csv')
columns = ['Gender', 'Age', 'Distance', 'PD1', 'ED1', 'HH_Vehicles', 'HH_inc', 'av_metro', 'tt', 'tc', 'Mode']
df1 = df[columns]
df_filtered = df1[df1['Mode'].isin([2, 5, 8])]

X = df_filtered.drop(columns=['Mode'])
y = df_filtered['Mode']

# Apply t-SNE
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)
# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plots 
ax.scatter(X_tsne[y == 2, 0], X_tsne[y == 2, 1], X_tsne[y == 2, 2], label='Bus', color='red', s=50, alpha=0.6)
ax.scatter(X_tsne[y == 5, 0], X_tsne[y == 5, 1], X_tsne[y == 5, 2], label='2w', color='green', s=50, alpha=0.6)
ax.scatter(X_tsne[y == 8, 0], X_tsne[y == 8, 1], X_tsne[y == 8, 2], label='walk', color='blue', s=50, alpha=0.6)

# Set labels for the three axes
ax.set_xlabel('tsne 1')
ax.set_ylabel('tsne 2')
ax.set_zlabel('tsne 3')

# Add a legend
ax.legend()

# Show the 3D scatter plot
plt.show()

#################################################
################################################
from sklearn.decomposition import PCA

# Apply PCA for 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create a 3D scatter plot for PCA
fig_pca = plt.figure(figsize=(12, 8))
ax_pca = fig_pca.add_subplot(111, projection='3d')

# Scatter plots with larger, semi-transparent markers
ax_pca.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], X_pca[y == 2, 2], label='Bus', color='red', s=50, alpha=0.6)
ax_pca.scatter(X_pca[y == 5, 0], X_pca[y == 5, 1], X_pca[y == 5, 2], label='2w', color='green', s=50, alpha=0.6)
ax_pca.scatter(X_pca[y == 8, 0], X_pca[y == 8, 1], X_pca[y == 8, 2], label='walk', color='blue', s=50, alpha=0.6)

# Set labels for the three axes
ax_pca.set_xlabel('PCA Component 1')
ax_pca.set_ylabel('PCA Component 2')
ax_pca.set_zlabel('PCA Component 3')

# Add a legend
ax_pca.legend()

# Show the 3D scatter plot for PCA
plt.show()

############################################
import umap

# Apply UMAP for 3 components
umap_model = umap.UMAP(n_components=3)
X_umap = umap_model.fit_transform(X)

# Create a 3D scatter plot for UMAP
fig_umap = plt.figure(figsize=(12, 8))
ax_umap = fig_umap.add_subplot(111, projection='3d')

# Scatter plots with larger, semi-transparent markers
ax_umap.scatter(X_umap[y == 2, 0], X_umap[y == 2, 1], X_umap[y == 2, 2], label='Bus', color='red', s=50, alpha=0.6)
ax_umap.scatter(X_umap[y == 5, 0], X_umap[y == 5, 1], X_umap[y == 5, 2], label='2w', color='green', s=50, alpha=0.6)
ax_umap.scatter(X_umap[y == 8, 0], X_umap[y == 8, 1], X_umap[y == 8, 2], label='walk', color='blue', s=50, alpha=0.6)

# Set labels for the three axes
ax_umap.set_xlabel('UMAP Component 1')
ax_umap.set_ylabel('UMAP Component 2')
ax_umap.set_zlabel('UMAP Component 3')

# Add a legend
ax_umap.legend()

# Show the 3D scatter plot for UMAP
plt.show()