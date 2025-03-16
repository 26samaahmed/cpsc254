from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import numpy as np


iris = load_iris()
X = iris.data  
true_labels = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
cluster_centers = kmeans.cluster_centers_
distances = np.linalg.norm(X_scaled - cluster_centers[y_kmeans], axis=1)
rmse = np.sqrt(np.mean(distances**2))
print(f'RMSE: {rmse:.2f}')

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Transform cluster centers to PCA space
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

# Plot the PCA-transformed data with K-Means clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5, edgecolors='k', label="Data Points")
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], c='red', s=300, alpha=0.75, marker='X', label="Cluster Centers")
plt.title('K-Means Clustering on Iris Dataset (PCA Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

print(f'Cluster Centers (Original Space): \n{kmeans.cluster_centers_}')

# Compare Clustering with Actual Labels
cluster_quality = adjusted_rand_score(true_labels, y_kmeans)
print(f'Clustering Alignment with True Labels (Adjusted Rand Index): {cluster_quality:.2f}')
