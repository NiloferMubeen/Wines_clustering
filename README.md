# Wines_clustering
This project implements a data-driven algorithm to group wines into similar clusters based on a matrix of numerical values. The algorithm follows a structured approach using weighted Euclidean distance to identify groups of wines that share similar characteristics.
# Project Overview
The task is to create an algorithm capable of clustering wines based on numerical data. This project does not use traditional clustering methods (like K-Means) but implements a custom algorithm that uses weighted Euclidean distance to group wines. The goal is to group similar wines together.

# Key Components
### 1. Matrix Class:

* A custom matrix class is used to handle the dataset and perform various operations such as loading from CSV, standardizing, and calculating Euclidean distances.

### 2. Custom Clustering Algorithm:

* A unique algorithm is used to group the wines into clusters based on their similarity. The algorithm involves initializing centroids, calculating weighted distances, and updating centroids and weights iteratively.
  
### 3. Data Standardization:

* The dataset is standardized to ensure consistency when calculating distances. The algorithm is then applied to this standardised data

### 4. Functions:

* Various helper functions are implemented to handle operations such as generating random weights, computing centroids, and calculating within-cluster and between-cluster separation.

# Tools used
* Python
* Pandas
* numpy

