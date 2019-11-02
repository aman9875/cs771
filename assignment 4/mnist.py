import numpy as np
import  matplotlib.pyplot as plt
import scipy.io
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pickle

file = open("embeddings.obj",'r')
X_embedded = pickle.load(file)
print(X_embedded[:,1])

kmeans = KMeans(n_clusters = 8,init='random').fit(X_embedded)
for i in set(kmeans.labels_):
	index = kmeans.labels_ == i
	plt.plot(X_embedded[index,0], X_embedded[index,1], 'o', markersize=3)
plt.show()
