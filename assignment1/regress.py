import numpy as np

num_seen_classes = 40
num_unseen_classes = 10
num_features = 4096
num_test_examples = 6180
lambda_values = [0.01,0.1,1,10,20,50,100]

# function to calculate the unseen class whose mean is at minimum distance from the mean of a given test sample.
def calc_min_mean(mean_unseen,x_i):
	distance = np.sum((mean_unseen - x_i)**2 , axis = 1)
	min_index = np.argmin(distance)
	return min_index   

#loading data
X_seen = np.load('X_seen.npy')
Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')
class_attributes_seen=np.load('class_attributes_seen.npy')
class_attributes_unseen=np.load('class_attributes_unseen.npy')

mean_seen = np.zeros((num_seen_classes,num_features))
mean_unseen = np.zeros((num_unseen_classes,num_features))

#calculating mean of seen classes
for i in range(num_seen_classes):
	mean_seen[i] = np.mean(X_seen[i] , axis = 0)

identity_matrix = np.identity(class_attributes_seen.shape[1])
Y_predict = np.zeros((num_test_examples))
max_accuracy = 0
best_lambda = 0

for lambda_value in lambda_values:
	#calculating mean for unseen classes using similarity
	inverse = np.linalg.inv(np.dot(class_attributes_seen.T , class_attributes_seen) + lambda_value*identity_matrix)
	w = np.dot(np.dot(inverse , class_attributes_seen.T) , mean_seen)
	mean_unseen = np.dot(class_attributes_unseen,w)

	#calculating prediction for each test example
	for i in range(num_test_examples):
		min_index = calc_min_mean(mean_unseen,Xtest[i])
		Y_predict[i] = min_index + 1

	#calculating accuracy
	count = 0
	for i in range(num_test_examples):
		if(Ytest[i] == Y_predict[i]):
			count += 1	

	accuracy = count/(num_test_examples*1.0)
	if(accuracy > max_accuracy):
		max_accuracy = accuracy
		best_lambda = lambda_value

	print("Accuracy = %f 		lambda = %f" %(accuracy,lambda_value)) 

print("Maximum accuracy = %f 	Optimal lambda = %f" %(max_accuracy,best_lambda))
