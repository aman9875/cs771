import numpy as np
import  matplotlib.pyplot as plt
import scipy.io
import pickle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

mat = scipy.io.loadmat('mnist_small.mat')
print(mat['X'].shape)

N = 10000
D = 784

X = mat['X']
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

f = open(b"embeddings_tsne.obj","wb")
pickle.dump(X_embedded,f)