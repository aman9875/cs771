import numpy as np
import  matplotlib.pyplot as plt
import csv
import  math

x_train = []  #input data
y_train = []  #output data

gamma = 0.1
lambda_value = 100             #regularization hyperparameter

with open('ridgetrain.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x_train.append(float(a))
		y_train.append(float(b))

#Calculating kernel matrix
n = len(x_train)
k = np.zeros((n,n))
for i in range(0,n):
	for j in range(0,n):
		k[i][j] = math.exp((-1) * gamma * (x_train[i]-x_train[j]) * (x_train[i]-x_train[j]))

#evaluating matrices for kennelized linear regression
identity_matrix = np.identity(n)
temp_matrix = k + lambda_value*identity_matrix 
alpha = np.dot(np.linalg.inv(temp_matrix),y_train)

x_test = []
y_test = []

with open('ridgetest.txt' , 'r') as f:
	for row in f:
		a , b = row.split()
		x_test.append(float(a))
		y_test.append(float(b))

#calculating the predictions
m = len(x_test)
y_pred = np.zeros(m)
for i in range(0,m):
	for j in range(0,n):
		y_pred[i] = y_pred[i] + alpha[j]*math.exp((-1)*gamma*(x_train[j]-x_test[i])*(x_train[j]-x_test[i]))

rmse_value = np.sum((y_test-y_pred)*(y_test-y_pred))
print(rmse_value)

plt.plot(x_test,y_test,'bo')
plt.plot(x_test,y_pred,'ro')
plt.show()