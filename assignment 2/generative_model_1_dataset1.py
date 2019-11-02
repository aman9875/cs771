import numpy as np
import  matplotlib.pyplot as plt
import csv
import math

x1 = []
x2 = []
y = []

num_examples = len(x1)

with open('binclass.txt' , 'rt') as f:
	reader  = csv.reader(f)

	for row in reader:
		x1.append(row[0])
		x2.append(row[1])
		y.append(row[2])

length = len(x1)
x = np.zeros((length,2))
x[:,0] = x1;
x[:,1] = x2;

print(x.shape)

x_plus = x[:200,:]
x_minus= x[200:,:]

mu_plus_optimal = np.sum(x_plus,axis = 0)/(x_plus.shape[0])
mu_minus_optimal = np.sum(x_minus,axis = 0)/(x_minus.shape[0])

print(mu_plus_optimal.shape)
print(mu_minus_optimal.shape)
print(x_plus.shape)
print(x_minus.shape)


sigma_plus_optimal = 0
sigma_minus_optimal= 0

for i in range(0,x_plus.shape[0]):
	x_1 = x_plus[i] 
	sigma_plus_optimal += np.dot((x_1-mu_plus_optimal),(x_1-mu_plus_optimal).T)

for i in range(0,x_minus.shape[0]):
	x_2 = x_minus[i]
	sigma_minus_optimal += np.dot((x_2-mu_minus_optimal),(x_2-mu_minus_optimal).T)

sigma_plus_optimal /= x_plus.shape[0]
sigma_minus_optimal /= x_minus.shape[0]


print(sigma_plus_optimal)
print(sigma_minus_optimal)

h = 0.5
x1_min, x1_max = x[:,0].min()- 1 , x[:,0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

print(xx.shape)
print(yy.shape)

z = np.c_[xx.ravel(),yy.ravel()]
print(z.shape)


x_r = []
y_r = []
x_b = []
y_b = []

for point in z:
	probability1 = math.exp(((-1)*np.dot((point-mu_plus_optimal),
		(point-mu_plus_optimal).T))/(2*sigma_plus_optimal))/(2*np.pi*np.sqrt(sigma_plus_optimal))

	probability2 = math.exp(((-1)*np.dot((point-mu_minus_optimal),
		(point-mu_minus_optimal).T))/(2*sigma_minus_optimal))/(2*np.pi*np.sqrt(sigma_minus_optimal))

	if(probability1 > probability2):
		x_r.append(point[0])
		y_r.append(point[1])
	else:
		x_b.append(point[0])
		y_b.append(point[1])

plt.plot(x_r,y_r,'ro')
plt.plot(x_b,y_b,'bo')

plt.show()