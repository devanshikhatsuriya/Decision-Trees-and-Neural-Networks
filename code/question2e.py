#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import matplotlib.pyplot as plt
import time
from question2b import neural_network, confusion_matrix


import sys


# In[2]:


# there are 10 categorical attributes and 1 class (ordinal 0-9)
column_names = ["a"+str(i) for i in range(10)]
column_names.append("class")

# In[2]:

if len(sys.argv) != 3:
	print("Please provide relative or absolute <path_of_train_data> and <path_of_test_data> as command line arguments.")
	sys.exit()
train_path = sys.argv[1]
test_path = sys.argv[2]

# train_df = pandas.read_csv("../poker_dataset/poker-hand-training-true.data", names=column_names)
# test_df = pandas.read_csv("../poker_dataset/poker-hand-testing.data", names=column_names)
try:
	train_df = pandas.read_csv(train_path, names=column_names)
	test_df = pandas.read_csv(test_path, names=column_names)
except:
	print("Error: Incorrect path for data")
	sys.exit()


print("Encoding attributes as one-hot-encoding...")

# convert attributes to one-hot encoding
attr_cols = column_names[:-1]
prefixes = attr_cols
train_df = pandas.get_dummies(train_df, prefix=prefixes, columns=attr_cols) # attribute columns converted to one-hot encodings
test_df = pandas.get_dummies(test_df, prefix=prefixes, columns=attr_cols) # attribute columns converted to one-hot encodings

# move class label to the last column
class_labels = train_df['class']
train_df = train_df.drop('class', axis=1)
train_df = train_df.join(class_labels)
class_labels = test_df['class']
test_df = test_df.drop('class', axis=1)
test_df = test_df.join(class_labels)


# In[6]:


train_data = train_df.to_numpy()
train_X = train_data[:, :-1]
intercept = numpy.ones((train_X.shape[0], 1))
train_X = numpy.append(intercept, train_X, axis=1)
train_y = train_data[:, -1]
test_data = test_df.to_numpy()
test_X = test_data[:, :-1]
intercept = numpy.ones((test_X.shape[0], 1))
test_X = numpy.append(intercept, test_X, axis=1)
test_y = test_data[:, -1]


# In[7]:


learning_rate = 0.1
M = 100
adaptive = True
hl_architecture = [100, 100]
relu = False
    
print(f"\nUsing hidden layer architecture {hl_architecture}...")

print(f"\nUsing sigmoid activation...")
nn_model = neural_network(M, 85, hl_architecture, 10, learning_rate, adaptive, relu)

print("\tTraining neural network model...")
start = time.process_time()
nn_model.train(train_df)
time_taken = time.process_time()-start
print(f"\tTime taken: {time_taken} seconds")

train_acc = nn_model.compute_accuracy(train_X, train_y)
test_acc = nn_model.compute_accuracy(test_X, test_y)
print(f"\tTrain accuracy: {train_acc}")
print(f"\tTest accuracy: {test_acc}")

cm = confusion_matrix(10, test_y, nn_model.predict(test_X))
print("\tConfusion matrix:\n", cm)


# In[9]:


relu = True

print(f"\nUsing relu activation for hidden layers...")
nn_model = neural_network(M, 85, hl_architecture, 10, learning_rate, adaptive, relu)

print("\tTraining neural network model...")
start = time.process_time()
nn_model.train(train_df)
time_taken = time.process_time()-start
print(f"\tTime taken: {time_taken} seconds")

train_acc = nn_model.compute_accuracy(train_X, train_y)
test_acc = nn_model.compute_accuracy(test_X, test_y)
print(f"\tTrain accuracy: {train_acc}")
print(f"\tTest accuracy: {test_acc}")

cm = confusion_matrix(10, test_y, nn_model.predict(test_X))
print("\tConfusion matrix:\n", cm)





