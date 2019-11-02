import numpy as np
import  matplotlib.pyplot as plt
import scipy.io
from sklearn.cluster import KMeans
import pickle


file = open("embeddings_pca.obj",'r')
X_embedded = pickle.load(file)

for it in range(10):
	kmeans = KMeans(n_clusters = 10,init='random',n_init = 1).fit(X_embedded)
	for i in set(kmeans.labels_):
		index = kmeans.labels_ == i
		plt.plot(X_embedded[index,0], X_embedded[index,1], 'o', markersize=3)
	plt.show()