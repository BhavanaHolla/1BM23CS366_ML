# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# =========================
# STEP 1: LOAD DATASET FROM URL
# =========================
# Mall Customers dataset (commonly used for clustering)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print(df.head())

# =========================
# STEP 2: SELECT FEATURES
# =========================
# Take only 2 features for easy visualization
X = df[['sepal_length', 'sepal_width']]

# =========================
# STEP 3: APPLY K-MEANS
# =========================
# n_clusters = number of groups we want
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit model
kmeans.fit(X)

# Predict cluster labels
labels = kmeans.labels_

# =========================
# STEP 4: METRICS
# =========================
# Inertia = sum of squared distances (lower is better)
print("Inertia:", kmeans.inertia_)

# =========================
# STEP 5: VISUALIZATION
# =========================
# Plot data points with cluster colors
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
