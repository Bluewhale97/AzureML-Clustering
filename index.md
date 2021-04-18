## Introduction

Clustering is a form of unsupervised learning. Like G means and K means clustering, they are commonly used in what we said, exploratory data analysis, when we want to get more intuition from exploratroy. The clustering models seek to identify groups and clusters of entities and separate them from other clusters.

## 1. Prerequisition

The seeds dataset this part used is from the UCI dataset repository(Dua, D. and Graff, C.(2019). 

Now we are going to read data:

```python
import pandas as pd

# load the training dataset
data = pd.read_csv('data/seeds.csv')

# Display a random sample of 10 observations (just the features)
features = data[data.columns[0:6]]
features.sample(10)
```

There are 6 features for each row:

![image](https://user-images.githubusercontent.com/71245576/115155152-986a6400-a04c-11eb-8efd-831bcfc6de9f.png)

In the tutorial, it suggests that because of six-dimensional space is difficult to visualize in a three dimensional world, we can use PCA here:

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Normalize the numeric features so they're on the same scale
scaled_features = MinMaxScaler().fit_transform(features[data.columns[0:6]])

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
features_2d[0:10]
```

Now that we have features translated to two dimensions, visualize them in a plot:

```python
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(features_2d[:,0],features_2d[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()
```

The data of two dimensions: 

![image](https://user-images.githubusercontent.com/71245576/115155235-f4cd8380-a04c-11eb-9bba-336efa8466ec.png)

Intuitively, there are at least two or three distinct groups of data points. The problem here is that without known class labels, how do you know how many clusters to separate the data into?

One way to try is to use a data sample to create a series of clustering models with an incrementing number of clusters and measure how tightly the data points are grouped within each cluster. A metric often used to measure this tightness is the within cluster sum of squares(WCSS).

```python
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

# Create 10 models with 1 to 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    # Fit the data points
    kmeans.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmeans.inertia_)
    
#Plot the WCSS values onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
The plot shows a large reduction in WCSS (so greater tightness) as the number of clusters increases from one to two, and a further noticable reduction from two to three clusters. After that, the reduction is less pronounced, resulting in an "elbow" in the chart at around three clusters. This is a good indication that there are two to three reasonably well separated clusters of data points.

![image](https://user-images.githubusercontent.com/71245576/115155326-6efe0800-a04d-11eb-9673-ff9843bb159a.png)

## 2. K-means clustering

K-means clustering is a commonly used algorithm which separates the data into K clusters. How it performs will be discussed in another article, we should know now is that there are K clusters separated and  clusters are formed by assigning the data points to their closest centroid. Now let's try using K means on our data set with a K of 3. 

```python
from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters
```

Let's visualize these cluster assignments with the two-dimensional data points:

```python
def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)
```

The data has been separated into three distinct clusters:

![image](https://user-images.githubusercontent.com/71245576/115155562-96a1a000-a04e-11eb-8340-d0e77167b38a.png)

In the case of the seeds data, the different species of seed are already known and encoded as 0 (Kama), 1 (Rosa), or 2 (Canadian), so we can use these identifiers to compare the species classifications to the clusters identified by our unsupervised algorithm

```python
seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)
```
There may be some differences between the cluster assignments and class labels, but the K-Means model should have done a reasonable job of clustering the observations so that seeds of the same species are generally in the same cluster.

![image](https://user-images.githubusercontent.com/71245576/115155649-ee400b80-a04e-11eb-8201-15f8b576547e.png)

## 3. Hierarchical clustering

Hierarchical clustering methods make fewer distributional assumptions when compared to K-means methods, which create clusters by either a divisive method(top down) or agglomerative method(bottom up). 

Let's see an example of clustering using an agglomerative clustering algorithm:

```python
from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters
```

Visualize the cluster assignments:
```python
import matplotlib.pyplot as plt

%matplotlib inline

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, agg_clusters)
```

The visualization shows:

![image](https://user-images.githubusercontent.com/71245576/115155799-7c1bf680-a04f-11eb-8d44-7bf97c1dd68a.png)

## Reference

Train and evaluate clustering models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-cluster-models/
