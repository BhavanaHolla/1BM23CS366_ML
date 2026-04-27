# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# STEP 1: LOAD DATASET FROM URL
# =========================
# Using Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print(df.head())

# =========================
# STEP 2: SELECT FEATURES
# =========================
# Drop target column (species)
X = df.drop("species", axis=1)

# =========================
# STEP 3: STANDARDIZE DATA
# =========================
# PCA is sensitive to scale → scaling is important
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# STEP 4: APPLY PCA
# =========================
# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# STEP 5: IMPORTANT OUTPUTS
# =========================
# Explained variance → how much information retained
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# =========================
# STEP 6: VISUALIZATION
# =========================
# Plot reduced 2D data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Projection")
plt.show()
