import csv
import numpy as np 
from scipy.spatial.distance import squareform

def load_data(filepath):
    data = []
    with open(filepath , 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Keep only information after the second column
            modified_row = {key: value.strip() for key, value in row.items() if key != ''}
            data.append(modified_row)
    return data

def calc_features(row):
    feats = []
    feats = [float(value) for key, value in row.items() if key != 'Country']
    #print(feats)
    return np.array(feats, dtype= np.float64)

def calculate_distance(features):
    distances = np.zeros((len(features), len(features)))
    for i in range (len(features)):
        for j in range(len(features)):
            distance = np.linalg.norm(features[i] - features[j])
            distances[i][j] = distance
    return distances

def merge(Z, dataset, clusters, n, idx_i, idx_j, dist):
    for i in clusters:
        idx = i[0]
        if idx == idx_i:
            cluster_i = i
        if idx == idx_j:
            cluster_j = i
    m = len(dataset)
    cluster1 = list(cluster_i[1])
    cluster2 = list(cluster_j[1])
    
    cluster1.extend(cluster2)
    
    minn=0
    maxx=0
    if idx_i <idx_j:
        minn= idx_i
    else:
        minn= idx_j
    
    if idx_i >idx_j:
        maxx= idx_i
    else:
        maxx=idx_j
        
    Z.append([minn, maxx, dist, len(cluster1)]) 
    
    nc = [m+n, tuple(cluster1)]
    clusters.remove(cluster_i)
    clusters.remove(cluster_j)
    clusters.append(nc)
    return Z, clusters



def calc_cluster(clusters,distances):
    answer_i = -1
    answer_j = -1
    mini = 10**9
    for i in clusters:
        idx_i = i[0]
        cli_i=i[1]
        for j in clusters:
            idx_j = j[0]
            cli_j = j[1]
            if idx_i == idx_j:
                continue             
            maxi = -1
            psuedo_i = -1
            psuedo_j = -1
            for k in cli_i:
                for l in cli_j:
                    if distances[k][l] > maxi:
                        maxi = float(distances[k,l])
                        psuedo_i = k
                        psuedo_j = l 
            if mini > maxi and maxi >= 0:
                mini = maxi
                answer_i = idx_i
                answer_j = idx_j 
    return [answer_i,answer_j,mini]         


import numpy as np

def hac(features):
    n = len(features)
    distance_matrix = np.zeros((n, n))  # Initialize distance matrix

    # Fill distance matrix with Euclidean distances
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # Initialize clustering array
    Z = np.zeros((n - 1, 4))

    # Helper function to find the indices of the minimum distance in the distance matrix
    def find_min_distance(matrix):
        min_distance = np.inf
        min_indices = (0, 0)

        for i in range(len(matrix)):
            for j in range(i+1, len(matrix[i])):
                if matrix[i, j] < min_distance:
                    min_distance = matrix[i, j]
                    min_indices = (i, j)

        return min_indices

    # HAC algorithm
    current_clusters = set(range(n))
    next_cluster_index = n

    for i in range(n - 1):
        # Find closest clusters using complete linkage and apply tie-breaking rule
        min_i, min_j = find_min_distance(distance_matrix)

        # Update Z[i, 0], Z[i, 1], Z[i, 2], Z[i, 3] based on the closest clusters
        Z[i, 0] = min_i
        Z[i, 1] = min_j
        Z[i, 2] = distance_matrix[min_i, min_j]
        Z[i, 3] = len(current_clusters)

        # Update the distance matrix by merging clusters min_i and min_j
        new_cluster_distances = np.maximum(distance_matrix[min_i, :], distance_matrix[min_j, :])
        distance_matrix = np.delete(distance_matrix, [min_i, min_j], axis=0)
        distance_matrix = np.delete(distance_matrix, [min_i, min_j], axis=1)
        new_cluster_distances = np.append(new_cluster_distances, 0)
        distance_matrix = np.concatenate([distance_matrix, new_cluster_distances.reshape(-1, 1)], axis=1)
        distance_matrix = np.concatenate([distance_matrix, new_cluster_distances.reshape(1, -1)], axis=0)
        new_cluster_distances = np.append(new_cluster_distances, 0)
        distance_matrix = np.column_stack([distance_matrix, new_cluster_distances])

        # Update the indices of the merged clusters
        current_clusters.remove(min_i)
        current_clusters.remove(min_j)
        current_clusters.add(next_cluster_index)

        # Increment the cluster index for the next iteration
        next_cluster_index += 1

    return Z

# Example usage:
# result = hac(features)

# Example usage (replace with your actual data)

data = load_data("countries.csv")
print(len(data))

features = [calc_features(row) for row in data]
print(hac(features))



