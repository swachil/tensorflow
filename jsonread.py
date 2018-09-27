# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:32:40 2018

@author: prasann
"""

import json
import numpy as np
import tensorflow as tf
#from functions import create_samples

with open('health.json') as json_file:
  x_data = json.load(json_file)
  #print(x_data[2])

def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples

items = []
for item in x_data:
    items.append(item['fields']['value'])
print(len(items))

n_features = 2
n_clusters = 5
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

model = tf.global_variables_initializer()
centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
print(samples)
print(centroids)

with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)
    
print(sample_values)
print(centroid_values)

def plot_clusters(all_samples, centroids, n_samples_per_cluster):
     import matplotlib.pyplot as plt
    #Plot out the different clusters
     #Choose a different colour for each cluster
     colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
     for i, centroid in enumerate(centroids):
         #Grab just the samples fpr the given cluster and plot them out with a new colour
         samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
         plt.scatter(samples[:,0], samples[:,1], c=colour[i])
         #Also plot centroid
         plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
         plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
     plt.show()
     
plot_clusters(sample_values, centroid_values, n_samples_per_cluster)