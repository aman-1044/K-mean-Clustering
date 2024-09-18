import pandas as pd

# Load the dataset
file_path = 'D:\Python\Mall_Customers (1).csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Selecting the features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Using the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-cluster sum of square
plt.show()

# Applying KMeans to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Adding the cluster labels to the original dataset
data['Cluster'] = y_kmeans

# Display the first few rows of the dataset with cluster labels
data.head()
