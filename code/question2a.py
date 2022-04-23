#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import matplotlib.pyplot as plt
import time

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

# In[3]:


print(train_df)


# In[4]:


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

print(train_df)
print(test_df)


# In[ ]:




