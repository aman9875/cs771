import numpy as np
import  matplotlib.pyplot as plt
import scipy.io
import pickle

mat = scipy.io.loadmat('mnist_small.mat')

K = 2
D = 784
N = 10000

X = mat['X'] #N*D matrix
Z = np.random.rand(N,K) #N*K matrix,initialize with random values
W = np.zeros((D,K)) #D*K matrix
print(X.shape)
print(Z.shape)
print(W.shape)

num_iterations = 0
while(num_iterations<500):
	W = (np.dot(np.linalg.inv(np.dot(Z.T,Z)),np.dot(Z.T,X))).T
	Z = np.dot(np.dot(X,W),np.linalg.inv(np.dot(W.T,W)))
	num_iterations += 1


f = open(b"embeddings_pca.obj","wb")
pickle.dump(Z,f)
