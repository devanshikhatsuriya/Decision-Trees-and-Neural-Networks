#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas
import numpy
import time
import sklearn


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


# In[28]:


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


# In[29]:


train_data = train_df.to_numpy()
train_X = train_data[:, :-1]
train_y = train_data[:, -1]
test_data = test_df.to_numpy()
test_X = test_data[:, :-1]
test_y = test_data[:, -1]


# In[30]:


from sklearn.neural_network import MLPClassifier

print("Creating sklearn.neural_network.MLPClassifier model...")
nn_model = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu', solver='sgd', batch_size=100, learning_rate='adaptive', learning_rate_init=0.1, verbose=True)

print("\tFitting Model to training data...")
start = time.process_time()
nn_model.fit(train_X, train_y)
time_taken = time.process_time()-start
print(f"\tTime taken: {time_taken} seconds")


# In[31]:


print(f"Train Accuracy: {nn_model.score(train_X, train_y)}")
print(f"Test Accuracy: {nn_model.score(test_X, test_y)}")


# In[ ]:




