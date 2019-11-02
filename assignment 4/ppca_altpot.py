import numpy as np
import  matplotlib.pyplot as plt
import scipy.io

mat = scipy.io.loadmat('facedata.mat')

D = 4096
N = 165
K_values = [10,20,30,40,50,100]
for K in K_values:
	X = mat['X'] #N*D matrix
	Z = np.random.rand(N,K) #N*K matrix,initialize with random values
	W = np.zeros((D,K)) #D*K matrix
	print(X.shape)
	print(Z.shape)
	print(W.shape)

	num_iterations = 0
	while(num_iterations<100):
		W = (np.dot(np.linalg.inv(np.dot(Z.T,Z)),np.dot(Z.T,X))).T
		Z = np.dot(np.dot(X,W),np.linalg.inv(np.dot(W.T,W)))
		num_iterations += 1

	mu = np.mean(X,axis=0)
	print(mu.shape)
	 
	fig, axes = plt.subplots(5,2)
	indices = [0,40,80,120,160]
	t = 0
	for i in indices:
		x_n = mu + np.dot(W,Z[i])
		img = np.reshape(x_n,(64,64))
		x_i = np.reshape(X[i],(64,64))
		axes[t,0].imshow(x_i.T,cmap='gray')
		axes[t,1].imshow(img.T,cmap='gray')
		plt.subplots_adjust(wspace=0,hspace=0)
		t += 1

	axes[4,0].set_xlabel("    Original Images             K = %d"%K)
	axes[4,1].set_xlabel("      Generated Images")
	plt.show()



