!pip install kagglehub
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("tawfikelmetwally/wine-dataset")

print("Path to dataset files:", path)
import os
31

print(os.listdir(path))
df = pd.read_csv(path + "/Wine dataset.csv")
df.head()
X = df.drop("class", axis=1) # features
y = df["class"] # target
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
import matplotlib.pyplot as plt

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Wine Dataset")
plt.show()
df.columns
