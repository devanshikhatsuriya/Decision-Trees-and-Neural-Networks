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
relu = False
hl_sizes = [5, 10, 15, 20, 25]
time_values = []
train_accuracies = []
test_accuracies = []

for hidden_layer_size in hl_sizes:
    
    print(f"\nUsing hidden layer size {hidden_layer_size}...")    
    nn_model = neural_network(M, 85, [hidden_layer_size], 10, learning_rate, adaptive, relu)
    
    print("\tTraining neural network model...")
    start = time.process_time()
    nn_model.train(train_df)
    time_taken = time.process_time()-start
    print(f"\tTime taken: {time_taken}")
    
    train_acc = nn_model.compute_accuracy(train_X, train_y)
    test_acc = nn_model.compute_accuracy(test_X, test_y)
    print(f"\tTrain accuracy: {train_acc}")
    print(f"\tTest accuracy: {test_acc}")
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    time_values.append(time_taken)   
    
    cm = confusion_matrix(10, test_y, nn_model.predict(test_X))
    print("\tConfusion matrix:\n", cm)
    


# In[10]:


plt.figure()
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.plot(hl_sizes, test_accuracies, label="Test Accuracy")
plt.plot(hl_sizes, train_accuracies, label="Train Accuracy")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Accuracies")
# plt.title(f"Accuracies with varying hidden size")
plt.legend()
plt.show()

fig_name = "2d_acc_plot.png"
fig.savefig(fig_name, dpi=100)

print(f"Plot saved as {fig_name}...")


# In[11]:


plt.figure(1)
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.plot(hl_sizes, time_values)
plt.xlabel("Hidden Layer Size")
plt.ylabel("Training time")
# plt.title(f"Training time with varying hidden size")
plt.show()

fig_name = "2d_time_plot.png"
fig.savefig(fig_name, dpi=100)

print(f"Plot saved as {fig_name}...")


# In[ ]:




