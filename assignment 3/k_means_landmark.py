import numpy as np
import  matplotlib.pyplot as plt
import csv
import  math
import random


x1 = []
x2 = []
L = 1
gamma = 0.1


def kernel(x1,x2):
	p = (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2
	return math.exp((-1)*gamma*p)

with open('kmeans_data.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x1.append(float(a))
		x2.append(float(b))

n = len(x1)
x = np.zeros((n,2))
x[:,0] = x1
x[:,1] = x2
r  = random.randint(0,n)
landmark_point = x[r]

#Calculating the feature vector using landmark point
X = np.zeros((n,L))
for i in range(0,n):
	X[i] = kernel(x[i],landmark_point)

mu = [0.0,0.0]
mu[0] = X[0]
mu[1] = X[1]
z = [0 for i in range(0,n)]
flag = 1
it = 0
while(flag==1 and it<=500):
	mu_new = [0,0]
	flag = 0
	#calculating the best cluster for each point
	z_new = [0 for i in range(0,n)]
	for i in range(0,n):
		z1 = (X[i]-mu[0])**2
		z2 = (X[i]-mu[1])**2
		if(z1<z2):
			z_new[i] = 0
		else:
			z_new[i] = 1

	#updating the means for the new cluster assignments
	c1 = 0
	c2 = 0
	for i in range(0,n):
		mu_new[z_new[i]] += X[i]
		if(z_new[i]==1):
			c1 += 1
		else:
			c2 += 1

	if(c1==0):
		mu_new[0] = 0
	else:
		mu_new[0] = mu_new[0]/c1

	if(c2==0):
		mu_new[1] = 0
	else:
		mu_new[1] = mu_new[1]/c2

	#checking for convergence
	for i in range(0,n):
		if(z[i]!=z_new[i]):
			flag = 1
			break

	mu[0] = mu_new[0]
	mu[1] = mu_new[1]
	z = z_new
	it += 1	

cluster1_x = []
cluster1_y = []
cluster2_x = []
cluster2_y = []

for i in range(0,n):
	if(z[i]==0):
		cluster1_x.append(x1[i])
		cluster1_y.append(x2[i])
	else:
		cluster2_x.append(x1[i])
		cluster2_y.append(x2[i])


plt.plot(cluster1_x,cluster1_y,'ro')
plt.plot(cluster2_x,cluster2_y,'go')
plt.plot(landmark_point[0],landmark_point[1],'bo')
plt.show()
