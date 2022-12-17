#Reference Page - https://github.com/endrol/DR_GCN/blob/9ad1929910ed30c3a623c25ba0da0198bd1655f5/dr_gcn/kmeans_feature_adj.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import pickle

#Loading the descriptors
sift_des = next(os.walk('/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess/sift_des'))[2]
sift_des.sort()


whole_sift = np.empty([0, 128])
length = []

#Appending the files for clustering
for desc in sift_des:
    test = np.load('/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess/sift_des/{}'.format(desc))
    print(desc," = ",test.shape)
    length.append(test.shape[0])
    whole_sift = np.append(whole_sift, test, 0)

## define the function to find correlation in one image
def all_np(arr):
    key = np.unique(arr)     #Considers only unique elements in the array
    result = {}              #Dictionary for storing the results
    for k in key:
        mask = (arr == k)    #Checking the number of occurences of a particular element
        arr_new = arr[mask]  
        v = arr_new.size     #Counting the occurences of an element
        result[k] = v        #Storing the aarray element as key and its occurence as value
    return result

## show the clustering info
n_classes = 20
n_samples = whole_sift.shape[0]
n_features = whole_sift.shape[1]
print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_classes, n_samples, n_features))

# K-Means Clustering
cluster = KMeans(n_clusters=20, init='k-means++', n_init=10) #20 clusters
cluster.fit(whole_sift)
#print("Feature_vector before normalizing = ",cluster.cluster_centers_)


# normalize feature vector and save as pickle file
gaussian_normalizer = StandardScaler()
feature_vector = gaussian_normalizer.fit_transform(cluster.cluster_centers_)
feature_tosave = open('description/node.pkl', 'wb')
pickle.dump(feature_vector, feature_tosave, -1)
feature_tosave.close()
print('feature_vector shape ',feature_vector.shape)


## define a dictionary structure for num and adj correlation and save pkl file
dr_adj = {}

# assume they are randomly distributed
dr_adj['nums'] = np.array([238,  243,    330,  181,  244,  186,  713,  337,  445,  141,  200,  421,  287,  245, 2008,  245,  96,  229,  261,  256])
adj_matrix = np.zeros([20, 20]) # 20 because the clusters are 20
start_point = 0
end_point = 0
counter = 0
for len in length:
    start_point = end_point
    end_point = len + start_point
    temp = cluster.labels_[start_point:end_point]
    dicti = all_np(temp)
    
    #Stores coocurences in an array
    add_vec = np.zeros([20])
    for key in dicti.keys():
        add_vec[key.item()] = dicti[key]

    #Using the coocurences to construct an adjacency matrix of 20 x 20
    for key in dicti.keys():
        adj_matrix[key.item()] = adj_matrix[key.item()] + add_vec
        adj_matrix[key.item()][key.item()] -= dicti[key]

dr_adj['adj'] = adj_matrix
print('show dr_adj, keys:{}\t value: dr_adj[nums]:{} \t dr_adj[adj]:{}\n'.format(dr_adj.keys(), dr_adj['nums'].shape, dr_adj['adj'].shape))
      
# save adj pickle file
adj_tosave = open('description/edge.pkl', 'wb')
pickle.dump(dr_adj, adj_tosave, -1)
adj_tosave.close()

