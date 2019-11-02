import numpy as np

num_seen_classes = 40
num_unseen_classes = 10
num_features = 4096
num_test_examples = 6180

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

#initializing and calculating similarity
similarity = np.zeros((num_unseen_classes,num_seen_classes))
similarity = np.dot(class_attributes_unseen , class_attributes_seen.T)
sum_similarity = np.sum(similarity , axis = 1)

#normalizing similarity
for i in range(num_unseen_classes):
	similarity[i] /= sum_similarity[i]

#calculating mean for unseen classes using similarity
for i in range(num_unseen_classes):
	for j in range(num_seen_classes):
		mean_unseen[i] += similarity[i][j] * mean_seen[j]

Y_predict = np.zeros((num_test_examples))

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
print("Accuracy = %f" %(accuracy))

