#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import time
import sklearn

import sys


# In[2]:

if len(sys.argv) != 4:
	print("Please provide relative or absolute <path_of_train_data>, <path_of_test_data> and <path_of_validation_data> as command line arguments.")
	sys.exit()

train_path = sys.argv[1]
test_path = sys.argv[2]
val_path = sys.argv[3]

# train_dataframe = pandas.read_csv('../bank_dataset/bank_train.csv', sep=';', header=0)
# test_dataframe = pandas.read_csv('../bank_dataset/bank_test.csv', sep=';', header=0) 
# val_dataframe = pandas.read_csv('../bank_dataset/bank_val.csv', sep=';', header=0)
try:
	train_dataframe = pandas.read_csv(train_path, sep=';', header=0)
	test_dataframe = pandas.read_csv(test_path, sep=';', header=0) 
	val_dataframe = pandas.read_csv(val_path, sep=';', header=0)
except:
	print("Error: Incorrect path for data")
	sys.exit()


# In[3]:


# print(train_dataframe.head())
# print(test_dataframe.head())
# print(val_dataframe.head())

# print(train_dataframe.shape)
# print(test_dataframe.shape)
# print(val_dataframe.shape)


# In[4]:


col_names = list(train_dataframe.columns) 
attr_names = col_names[:-1] # remove the prediction y column name
unique_values = [[] for i in range(len(col_names))]
n_attrs = len(attr_names)

for i in range(len(col_names)):
    unique_values[i] = sorted(train_dataframe[col_names[i]].unique())
    
attr_category = [] # 0 for real-valued, 1 for boolean, 2 for categorical
for i in range(len(attr_names)):
    if len(unique_values[i]) <= 2:
        attr_category.append(1) # boolean-valued attribute
    elif isinstance(unique_values[i][0], str):
        if len(unique_values[i]) <= 2:
            attr_category.append(1) # boolean-valued attribute
        else:
            attr_category.append(2) # categorical attribute
    else:
        attr_category.append(0) # real-valued attribute
        
print("Attributes:\n", attr_names)
print("\nCategorical Attributes:")
for i in range(1, len(attr_category)):
    c = attr_category[i]
    if c != 0:
        print(i, col_names[i], unique_values[i])
print("\nOrdinal Attributes:")
for i in range(1, len(attr_category)):
    c = attr_category[i]
    if c == 0:
        print(i, col_names[i])


# In[5]:


# convert categorical attributes with integer values to one-hot encodings
# convert yes/no attributes to 1/0
print("\nConverting categorical attributes to one-hot encodings and boolean attributes to 0/1...")
train_df_new = train_dataframe
test_df_new = test_dataframe
val_df_new = val_dataframe
for i in range(len(attr_names)):
    attr = attr_names[i]
    if attr_category[i] == 2:
        one_hot = pandas.get_dummies(train_df_new[attr], prefix=attr) # one-hot encodings
        train_df_new = train_df_new.drop(attr, axis=1)
        train_df_new = train_df_new.join(one_hot)
        one_hot = pandas.get_dummies(test_df_new[attr], prefix=attr) # one-hot encodings
        test_df_new = test_df_new.drop(attr, axis=1)
        test_df_new = test_df_new.join(one_hot)
        one_hot = pandas.get_dummies(val_df_new[attr], prefix=attr) # one-hot encodings
        val_df_new = val_df_new.drop(attr, axis=1)
        val_df_new = val_df_new.join(one_hot)
    elif attr_category[i] == 1:
        train_df_new[attr] = (train_df_new[attr] == unique_values[i][1]).astype(int)
        test_df_new[attr] = (test_df_new[attr] == unique_values[i][1]).astype(int)
        val_df_new[attr] = (val_df_new[attr] == unique_values[i][1]).astype(int)
# convert y labels from yes/no to 1/0
y = (train_df_new['y'] == unique_values[-1][1]).astype(int)
train_df_new = train_df_new.drop('y', axis=1)
train_df_new = train_df_new.join(y)
y = (test_df_new['y'] == unique_values[-1][1]).astype(int)
test_df_new = test_df_new.drop('y', axis=1)
test_df_new = test_df_new.join(y)
y = (val_df_new['y'] == unique_values[-1][1]).astype(int)
val_df_new = val_df_new.drop('y', axis=1)
val_df_new = val_df_new.join(y)


# In[6]:


# (a) n estimators (50 to 450 inrange of 100). 
# (b) max features (0.1 to 1.0 in range of 0.2) 
# (c) min samples split (2 to 10 in range of 2)


# In[7]:


train_df_new.head()


# In[8]:


data = train_df_new.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]
test_data = test_df_new.to_numpy()
test_X = test_data[:, 0:-1]
test_y = test_data[:, -1]
val_data = val_df_new.to_numpy()
val_X = val_data[:, 0:-1]
val_y = val_data[:, -1]


# In[9]:


from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(rf, param_grid, verbose=1)

param_grid = {'n_estimators': list(range(50, 451, 100)), 'max_features': [0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': list(range(2, 11, 2))}

best_oob_score = None
store = numpy.empty((3,))
best_rf = None

print("\nPerforming Grid Search for best hyperparameters...\n")

for n_estimators in param_grid['n_estimators']:
	for max_features in param_grid['max_features']:
		for min_samples_split in param_grid['min_samples_split']:
			print(f"\tCreating Random Forest CLassifier with n_estimators={n_estimators}, max_features={max_features}, min_samples_split={min_samples_split}...")
			rf = RandomForestClassifier(criterion="entropy", bootstrap=True, oob_score=True, verbose=1, n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)
			start = time.process_time()
			rf.fit(X, y)
			print(f"\tTime taken: {time.process_time() - start} seconds")
			score = rf.oob_score_
			print(f"\tOOB Score: {score}\n")
			if best_oob_score==None or score > best_oob_score:
				best_oob_score = score
				best_rf = rf
				store[0] = n_estimators
				store[1] = max_features
				store[2] = min_samples_split

# In[ ]:


# store best parameters
numpy.savetxt('1c_best_parameters.csv', store, delimiter=',')


# In[ ]:


print(f"\nBest OOB score: {best_oob_score}")
print(f"Best Parameters: {store}")
print(f"Train Accuracy: {best_rf.score(X, y)}")
print(f"Test Accuracy: {best_rf.score(test_X, test_y)}")
print(f"Validation Accuracy: {best_rf.score(val_X, val_y)}")


# In[ ]:




