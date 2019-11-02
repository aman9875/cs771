import numpy as np
import  matplotlib.pyplot as plt
import csv
import  math

x1 = []
x2 = []

with open('kmeans_data.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x1.append(float(a))
		x2.append(float(b))

epsilon = 0.01
n = len(x1)
x = np.zeros(n)

#We can convert each two dimensional point into a single dimensional 
#feature which represents the distance of the point from the origin
for i in range(0,n):
	x[i] = x1[i]*x1[i] + x2[i]*x2[i] 

mu = [0.0,0.0]
mu[0] = x[0]
mu[1] = x[1]
z = [0 for i in range(0,n)]
flag = 1
while(flag==1):
	mu_new = [0,0]
	flag = 0
	z_new = [0 for i in range(0,n)]
	for i in range(0,n):
		z_new[i] = np.argmin((x[i]-mu)*(x[i]-mu)) #calculating the best cluster for each point

	#updating the means for the new cluster assignments
	for i in range(0,n):
		mu_new[z_new[i]] += x[i]

	if(np.sum(z_new[i])!=0):
		mu_new[1] = mu_new[1]/(np.sum(z_new[i]));
		mu_new[0] = mu_new[0]/(n - np.sum(z_new[i]));
	else:
		mu_new[1] = 0
		mu_new[0] = mu_new[0]/n

	#checking for convergence
	for i in range(0,n):
		if(z[i]!=z_new[i]):
			flag = 1
			break

	mu[0] = mu_new[0]
	mu[1] = mu_new[1]
	z = z_new

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
plt.show()