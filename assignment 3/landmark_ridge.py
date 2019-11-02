import numpy as np
import  matplotlib.pyplot as plt
import csv
import  math

x_train = []
y_train = []
L = 100     #number of landmark points
gamma = 0.1
lambda_value = 0.1

with open('ridgetrain.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x_train.append(float(a))
		y_train.append(float(b))

n = len(x_train)
landmark_points = np.random.choice(x_train,L)

#Calculating the feature vector using landmark points
X = np.zeros((n,L))
for i in range(0,n):
	for j in range(0,L):
		X[i][j] = math.exp((-1)*gamma*(x_train[i]-landmark_points[j])*(x_train[i]-landmark_points[j]))

#calculating the weight matrix
temp_matrix = np.linalg.inv(np.dot(X.T,X) + lambda_value*np.identity(L))
w = np.dot(temp_matrix,np.dot(X.T,y_train))

x_test = []
y_test = []

with open('ridgetest.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x_test.append(float(a))
		y_test.append(float(b))

n = len(x_test)
X_test = np.zeros((n,L))

#Calculating the feature vector for test points
for i in range(0,n):
	for j in range(0,L):
		X_test[i][j] = math.exp((-1)*gamma*(x_test[i]-landmark_points[j])*(x_test[i]-landmark_points[j]))

y_pred = np.zeros(n)

for i in range(n):
	y_pred[i] =  np.dot(w.T,X_test[i])


rmse = np.sum((y_test-y_pred)*(y_test-y_pred))
print(rmse)

plt.plot(x_test,y_test,'bo')
plt.plot(x_test,y_pred,'ro')
plt.show()