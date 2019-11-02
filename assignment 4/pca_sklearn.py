from sklearn.decomposition import PCA

import numpy as np
import  matplotlib.pyplot as plt
import scipy.io
import pickle

mat = scipy.io.loadmat('mnist_small.mat')

K = 2
D = 784
N = 10000

X = mat['X'] #N*D matrix

pca = PCA(n_components = 2)
X_embedded = pca.fit_transform(X)

f = open(b"embeddings_pca.obj","wb")
pickle.dump(X_embedded,f)
