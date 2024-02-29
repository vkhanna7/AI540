import csv
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def load_data(filepath):
    data = []
    with open(filepath , 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            modified_row = {key: value for key, value in row.items()}
            data.append(modified_row)
    return data

def calc_features(row):
    data = [float(value) for key, value in row.items() if key != 'Country' and not(key == '')]
    X = np.array(data, dtype=np.float64)
    return X



def hac(dataset):
    Z = []  # To store the hierarchical clustering tree
    clusters = [(i, [i]) for i in range(len(dataset))]  # Initialize clusters

    # Calculate pairwise distances
    distances = np.zeros((len(dataset), len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            dist = np.linalg.norm(dataset[i] - dataset[j])
            distances[i][j] = dist

    for i in range(len(distances) - 1):
        idx_i, idx_j, dist, cluster_i, cluster_j = calc_cluster(clusters, distances)
        Z, clusters = merge(Z, dataset, clusters, i, idx_i, idx_j, dist, cluster_i, cluster_j)

    return np.array(Z).astype("float")


def calc_cluster(clusters, distances):
    answer_i = -1
    answer_j = -1
    mini = np.inf
    cluster_i = []
    cluster_j = []

    for i in clusters:
        idx_i, cli_i = i
        for j in clusters:
            idx_j, cli_j = j
            if idx_i == idx_j:
                continue
            maxi = -1
            for k in cli_i:
                for l in cli_j:
                    maxi = max(maxi, distances[k, l])
            if mini > maxi and maxi >= 0:
                mini = maxi
                answer_i = idx_i
                answer_j = idx_j
                cluster_i = cli_i
                cluster_j = cli_j

    return answer_i, answer_j, mini, cluster_i, cluster_j


def merge(Z, dataset, clusters, n, idx_i, idx_j, dist, cluster_i, cluster_j):
    m = len(dataset)
    cluster1 = list(cluster_i)
    cluster2 = list(cluster_j)

    cluster1.extend(cluster2)

    minn = min(idx_i, idx_j)
    maxx = max(idx_i, idx_j)

    Z.append([minn, maxx, dist, len(cluster1)])

    nc = [m + n, tuple(cluster1)]
    # Remove clusters based on content
    clusters = [c for c in clusters if c[0] not in (idx_i, idx_j)]
    clusters.append(tuple(nc))
    return Z, clusters

def distance_calc2(dataset): #distance matrix
    distances = np.zeros((len(dataset),len(dataset)))
    for i in range(len(dataset)):
        for j in range(len(dataset)):
                dist = np.linalg.norm(dataset[i] - dataset[j])
                distances[i][j] = dist
    return distances


def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 6))
    
    # Use dendrogram to visualize hierarchical clustering
    dn = hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)

    # Adjust layout to prevent x labels from being cut off
    plt.tight_layout()

    # Show the plot
    #plt.show()
    return fig

def normalize_features(features):
    # Convert the list of feature vectors to a NumPy array
    features_array = np.array(features)

    # Calculate mean and standard deviation for each column
    column_means = np.mean(features_array, axis=0)
    column_stddevs = np.std(features_array, axis=0)

    # Normalize each column in the feature vectors
    normalized_features = (features_array - column_means) / column_stddevs

    # Convert the result back to a list of NumPy arrays
    normalized_features_list = [np.array(row) for row in normalized_features]

    return normalized_features_list


data = load_data("countries.csv")
country_names = [row["Country"] for row in data]
features = [calc_features(row) for row in data]
features_normalized = normalize_features(features)
n = 51
Z_raw = hac(features[:n])
Z_normalized = hac(features_normalized[:n])
print(Z_normalized)
fig = fig_hac(Z_raw, country_names[:n])
plt.show()