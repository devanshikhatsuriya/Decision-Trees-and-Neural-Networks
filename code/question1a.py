#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import matplotlib.pyplot as plt
import time

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


# In[31]:


col_names = list(train_dataframe.columns) 
attr_names = col_names[:-1] # remove the prediction y column name
unique_values = [[] for i in range(len(attr_names))]
n_attrs = len(attr_names)

for i in range(len(attr_names)):
    unique_values[i] = sorted(train_dataframe[attr_names[i]].unique())
    
# print(unique_values)
    
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

print(f"Attributes: {attr_names}")             
# print(attr_category)
               


# In[5]:


class decision_tree_node:
    
    def __init__(self):
        self.root = None        
        self.split_attr = None# -- index of the attribute to split on
        self.split_attr_values = None # -- list of value(s) of the split attribute to split on
        self.parent = None
        self.children = []
        self.is_leaf = None
        self.y_value = None # -- value of prediction if the node is leaf
        self.is_empty = None # -- whether train dataframe here is empty
        self.dataframe = None # -- the train dataframe that was used to grow the tree at this node
        self.test_df = None # -- use to predict as the tree grows
        self.val_df = None # -- use to predict as the tree grows


# In[37]:


def compute_MI(dataframe, attr_no):
    # computes Mutual Information on split of data based on attribute attr_no.
    # For real-valued attributes, splits on median
    # For boolean-valued attributes, two way split
    # For categorical attributes, does multi-way split 
    # To use one-hot encodings for categorical attributes, pass dataframe after converting each categorical
    #     attribute to n boolean attributes
    
    if attr_category[attr_no] == 0:
        median = dataframe[col_names[attr_no]].median()
        set1 = dataframe[dataframe[col_names[attr_no]]<=median]
        set2 = dataframe[dataframe[col_names[attr_no]]>median]
        n1 = len(set1)
        total = len(dataframe)
        n2 = total - n1
        # what if n1 or n2 is 0?
        if n1 == 0:
            p0_set2 = len(set2[set2[col_names[-1]]=="no"])/n2
            p1_set2 = len(set2[set2[col_names[-1]]=="yes"])/n2
            if p0_set2 == 0:
                term2 = -1 * (p1_set2*numpy.log(p1_set2))
            elif p1_set2 == 0:
                term2 = -1 * (p0_set2*numpy.log(p0_set2))
            else:
                term2 = -1 * ((p0_set2*numpy.log(p0_set2)) + (p1_set2*numpy.log(p1_set2)))
            MI = -1 * ((n2/total)*term2)
        elif n2 == 0:
            p0_set1 = len(set1[set1[col_names[-1]]=="no"])/n1
            p1_set1 = len(set1[set1[col_names[-1]]=="yes"])/n1
            if p0_set1 == 0:
                term1 = -1 * (p1_set1*numpy.log(p1_set1))
            elif p1_set1 == 0:
                term1 = -1 * (p0_set1*numpy.log(p0_set1))
            else:
                term1 = -1 * ((p0_set1*numpy.log(p0_set1)) + (p1_set1*numpy.log(p1_set1)))
            MI = -1 * ((n1/total)*term1) # leaving out H(Y) term in H(Y)-H(Y|Xj)
        else:            
            p0_set1 = len(set1[set1[col_names[-1]]=="no"])/n1
            p1_set1 = len(set1[set1[col_names[-1]]=="yes"])/n1
            if p0_set1 == 0:
                term1 = -1 * (p1_set1*numpy.log(p1_set1))
            elif p1_set1 == 0:
                term1 = -1 * (p0_set1*numpy.log(p0_set1))
            else:
                term1 = -1 * ((p0_set1*numpy.log(p0_set1)) + (p1_set1*numpy.log(p1_set1)))
            p0_set2 = len(set2[set2[col_names[-1]]=="no"])/n2
            p1_set2 = len(set2[set2[col_names[-1]]=="yes"])/n2
            if p0_set2 == 0:
                term2 = -1 * (p1_set2*numpy.log(p1_set2))
            elif p1_set2 == 0:
                term2 = -1 * (p0_set2*numpy.log(p0_set2))
            else:
                term2 = -1 * ((p0_set2*numpy.log(p0_set2)) + (p1_set2*numpy.log(p1_set2)))
            MI = -1 * ((n1/total)*term1 + (n2/total)*term2) # leaving out H(Y) term in H(Y)-H(Y|Xj)
        return MI
    
    else: #  attr_category[attr_no] == 1 or attr_category[attr_no] == 2
        split_values = unique_values[attr_no]
        total = len(dataframe)
        MI = 0.0
        for value in split_values:
            cur_set = dataframe[dataframe[col_names[attr_no]]==value]
            n = len(cur_set)
            # what if n is 0?
            if n == 0:
                continue # MI += 0 * inf
            else:
                p0_set = len(cur_set[cur_set[col_names[-1]]=="no"])/n
                p1_set = len(cur_set[cur_set[col_names[-1]]=="yes"])/n
                if p0_set == 0:
                    MI += (n/total) * (p1_set*numpy.log(p1_set)) # minuses get cancelled
                elif p1_set == 0:
                    MI += (n/total) * (p0_set*numpy.log(p0_set)) # minuses get cancelled
                else:
                    MI += (n/total) * (p0_set*numpy.log(p0_set) + p1_set*numpy.log(p1_set)) # minuses get cancelled
        return MI
    


# In[38]:


def grow_tree(tree_node, dataframe):
    
    tree_node.dataframe = dataframe
    
    if dataframe.empty:
        tree_node.is_empty = True
        tree_node.is_leaf = True
        return
    else:
        tree_node.is_empty = False
    
    y_values_present = list(dataframe[col_names[-1]].unique())
    if len(y_values_present) == 1:
        tree_node.is_leaf = True
        tree_node.y_value = y_values_present[0]
        return
    else:
        tree_node.is_leaf = False     
    
    max_MI = None
    best_attr = None
    for j in range(n_attrs):            
        cur_MI = compute_MI(dataframe, j)
        if max_MI == None or cur_MI > max_MI:
            best_attr = j
            max_MI = cur_MI
        
    tree_node.split_attr = best_attr
    best_attr_name = col_names[best_attr]
    
    if attr_category[best_attr] == 0:
        median = dataframe[best_attr_name].median()
        tree_node.split_attr_values = [median]
        
        # what if left or right dataframes are empty        
        left_node = decision_tree_node()
        left_node.root = tree_node.root
        left_node.parent = tree_node        
        
        right_node = decision_tree_node()
        right_node.root = tree_node.root
        right_node.parent = tree_node
        
        tree_node.children.append(left_node)
        tree_node.children.append(right_node)       
        
        grow_tree(left_node, dataframe[dataframe[best_attr_name]<=median])
        grow_tree(right_node, dataframe[dataframe[best_attr_name]>median]) 
        
    else: # attr_category[best_attr] == 1 or attr_category[best_attr] == 2
        split_values = unique_values[best_attr]
        tree_node.split_attr_values = split_values
        for value in split_values:
            node = decision_tree_node()
            node.root = tree_node.root
            node.parent = tree_node
            tree_node.children.append(node)    
            # what if this dataframe is empty
            grow_tree(node, dataframe[dataframe[best_attr_name]==value])
            
    return


# In[8]:


print("\nMulti-way splitting for categorical attributes\n")

root_node = decision_tree_node()
root_node.root = root_node
print("Growing decision tree...")
start = time.process_time()
grow_tree(root_node, train_dataframe)
print(f"Time taken to grow decision tree: {time.process_time()-start} seconds")


# In[9]:


def prediction_helper(df, node, y_values):
        
    if node.is_empty:
        # print(f"Empty Node Encountered, {len(df)} examples skipped")
        return

    elif node.is_leaf:
        indices = numpy.array(df.index)
        y_values[indices] = node.y_value
        return

    else:
        split_attr_name = col_names[node.split_attr]
        if attr_category[node.split_attr] == 0:
            left_df = df[df[split_attr_name]<=node.split_attr_values[0]]
            right_df = df[df[split_attr_name]>node.split_attr_values[0]]
            prediction_helper(left_df, node.children[0], y_values)
            prediction_helper(right_df, node.children[1], y_values)
        else: # attr_category[node.split_attr] == 1 or attr_category[node.split_attr] == 2
            split_values = node.split_attr_values
            for i in range(len(split_values)):
                value = split_values[i]
                child_df = df[df[split_attr_name]==value]
                prediction_helper(child_df, node.children[i], y_values)

def prediction(test_X, root_node):
    
    y_values = numpy.empty(shape=(len(test_X)), dtype=object)        
    prediction_helper(test_X, root_node, y_values)
    return y_values
            


# In[10]:


train_prediction = prediction(train_dataframe.iloc[:,:-1], root_node)

train_accuracy = 100 * numpy.count_nonzero(train_prediction==train_dataframe['y'])/len(train_dataframe)
print("Train Accuracy:", train_accuracy)


# In[11]:


val_prediction = prediction(val_dataframe.iloc[:,:-1], root_node)
test_prediction = prediction(test_dataframe.iloc[:,:-1], root_node)

val_accuracy = 100 * numpy.count_nonzero(val_prediction==val_dataframe['y'])/len(val_dataframe)
print("Validation Accuracy:", val_accuracy)
test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_dataframe['y'])/len(test_dataframe)
print("Test Accuracy:", test_accuracy)


# In[13]:


def partial_accuracies(root_node, train_dataframe, test_dataframe, validation_dataframe):
    
    '''
        does bfs traversal of decision tree and computes train, test and validation accuracies
            as the tree grows
    '''
    
    bfs_tree = decision_tree_node()
    bfs_tree.root = bfs_tree
    bfs_tree.split_attr = root_node.split_attr
    bfs_tree.split_attr_values = root_node.split_attr_values
    bfs_tree.is_leaf = True
    if root_node.is_empty:
        bfs_tree.is_empty = True
    else:
        root_node.is_empty = False
        # find majority y in root_node.dataframe
        counts = dict(root_node.dataframe['y'].value_counts())
        if 'no' not in counts:
            bfs_tree.y_value = 'yes'                    
        elif 'yes' not in counts:
            bfs_tree.y_value = 'no'
        else:
            if counts['no'] >= counts['yes']:
                bfs_tree.y_value = 'no'
            else:
                bfs_tree.y_value = 'yes'
    bfs_tree.dataframe = root_node.dataframe
    bfs_tree.test_df = test_dataframe
    bfs_tree.val_df = validation_dataframe
    
    queue = []
    
    queue.append((bfs_tree, root_node))
    
    num_nodes = numpy.empty((0,))
    train_accs = numpy.empty((0,))
    test_accs = numpy.empty((0,))
    val_accs = numpy.empty((0,))
    
    leaves = set()
    leaves.add(bfs_tree)
    
    index = 0
    
    while len(queue) != 0:
        
        index += 1
        (node, dt_node) = queue.pop(0)
        
        # add node to partial tree
        if node.parent != None: # node is not root
            node.parent.children.append(node)
            if node.parent.is_leaf:
                node.parent.is_leaf = False
        
        # predict if parent node has completed its children   
        if node.parent == None or (len(node.parent.split_attr_values)==1 and len(node.parent.children)==2) or (len(node.parent.split_attr_values)>1 and len(node.parent.children)==len(node.parent.split_attr_values)):
            
            print(f"Predicting using {index} nodes...")
            
            if node.parent != None:
                # remove parent and add its children to prediction contour
                parent = node.parent
                leaves.remove(parent)
                for child in parent.children:
                    leaves.add(child)                
                                
            train_pred = numpy.empty(shape=(len(train_dataframe)), dtype=object)
            test_pred = numpy.empty(shape=(len(test_dataframe)), dtype=object)
            val_pred = numpy.empty(shape=(len(validation_dataframe)), dtype=object)
            
            for leaf in leaves:
                if not leaf.dataframe.empty:
                    indices = numpy.array(leaf.dataframe.index)
                    train_pred[indices] = leaf.y_value
                if not leaf.test_df.empty:
                    indices = numpy.array(leaf.test_df.index)
                    test_pred[indices] = leaf.y_value
                if not leaf.val_df.empty:
                    indices = numpy.array(leaf.val_df.index)
                    val_pred[indices] = leaf.y_value                
            
            train_acc = 100 * numpy.count_nonzero(train_pred==train_dataframe['y'])/len(train_dataframe)
            test_acc = 100 * numpy.count_nonzero(test_pred==test_dataframe['y'])/len(test_dataframe)
            val_acc = 100 * numpy.count_nonzero(val_pred==val_dataframe['y'])/len(validation_dataframe)        

            num_nodes = numpy.append(num_nodes, index)
            train_accs = numpy.append(train_accs, train_acc)
            test_accs = numpy.append(test_accs, test_acc)
            val_accs = numpy.append(val_accs, val_acc) 
        
        # add children of dt_node to queue
        for i in range(len(dt_node.children)):
            dt_node_child = dt_node.children[i]
            node_child = decision_tree_node()
            node_child.root = bfs_tree
            node_child.split_attr = dt_node_child.split_attr
            node_child.split_attr_values = dt_node_child.split_attr_values
            node_child.parent = node
            node_child.is_leaf = True
            if dt_node_child.is_empty:
                node_child.is_empty = True
            else:
                node_child.is_empty = False
                # find majority y in dt_node_child.dataframe
                counts = dict(dt_node_child.dataframe['y'].value_counts())
                if 'no' not in counts:
                    node_child.y_value = 'yes'                    
                elif 'yes' not in counts:
                    node_child.y_value = 'no'
                else:
                    if counts['no'] >= counts['yes']:
                        node_child.y_value = 'no'
                    else:
                        node_child.y_value = 'yes'
            node_child.dataframe = dt_node_child.dataframe
            # split test and validation dataframes
            split_attr_name = col_names[node.split_attr]
            if attr_category[node.split_attr] == 0:
                if i == 0:
                    node_child.test_df = node.test_df[node.test_df[split_attr_name]<=node.split_attr_values[0]]
                    node_child.val_df = node.val_df[node.val_df[split_attr_name]<=node.split_attr_values[0]]
                else:
                    node_child.test_df = node.test_df[node.test_df[split_attr_name]>node.split_attr_values[0]]
                    node_child.val_df = node.val_df[node.val_df[split_attr_name]>node.split_attr_values[0]]
            else: # attr_category[node.split_attr] == 1 or attr_category[node.split_attr] == 2
                split_values = node.split_attr_values
                value = split_values[i]
                node_child.test_df = node.test_df[node.test_df[split_attr_name]==value]
                node_child.val_df = node.val_df[node.val_df[split_attr_name]==value] 
            
            queue.append((node_child, dt_node_child))
            
    # print(num_nodes.shape, num_nodes)
    # print(train_accs.shape, train_accs)
    # print(test_accs.shape, test_accs)
    # print(val_accs.shape, val_accs)
            
    return numpy.concatenate(([num_nodes],[train_accs],[test_accs],[val_accs]),axis=0)


# In[14]:


print("Computing accuracies as decision tree grows...")
start = time.process_time()
acc = partial_accuracies(root_node, train_dataframe, test_dataframe, val_dataframe)
print(f"Time taken: {time.process_time()-start} seconds")


# In[15]:

plt.figure()
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.plot(acc[0,:], acc[1,:], label='Train Accuracy')
plt.plot(acc[0,:], acc[2,:], label='Test Accuracy')
plt.plot(acc[0,:], acc[3,:], label='Validation Accuracy')
plt.xlabel('No. of Nodes')
plt.ylabel('Accuracy (%)')

plt.legend()
plt.show()

fig_name = "1a_multiway.png"
fig.savefig(fig_name, dpi=100)
    
print(f"Plot saved as {fig_name}...")


# In[33]:


# Use one-hot encoding for categorical attributes
# transform each categorical attribute to n boolean attributes
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
# move y label to the last column
y = train_df_new['y']
train_df_new = train_df_new.drop('y', axis=1)
train_df_new = train_df_new.join(y)
y = test_df_new['y']
test_df_new = test_df_new.drop('y', axis=1)
test_df_new = test_df_new.join(y)
y = val_df_new['y']
val_df_new = val_df_new.drop('y', axis=1)
val_df_new = val_df_new.join(y)


# In[35]:


col_names = list(train_df_new.columns) 
attr_names = col_names[:-1] # remove the prediction y column name
unique_values = [[] for i in range(len(attr_names))]
n_attrs = len(attr_names)

for i in range(len(attr_names)):
    unique_values[i] = sorted(train_df_new[attr_names[i]].unique())
    
attr_category = [] # 0 for real-valued, 1 for boolean, 2 for categorical
for i in range(len(attr_names)):
    if len(unique_values[i]) <= 2:
        attr_category.append(1) # boolean-valued attribute
    else:
        attr_category.append(0) # real-valued attribute


# In[36]:


print("\nTwo-way split using boolean one-hot representation for categorical attributes\n")
print(f"Attributes: {attr_names}")
root_node = decision_tree_node()
root_node.root = root_node
print("Growing decision tree...")
start = time.process_time()
grow_tree(root_node, train_df_new)
print(f"Time taken to grow decision tree: {time.process_time()-start} seconds")


# In[39]:


print("Computing accuracies as decision tree grows...")
start = time.process_time()
acc = partial_accuracies(root_node, train_df_new, test_df_new, val_df_new)
print(f"Time taken: {time.process_time()-start} seconds")


# In[40]:

print("Generating accuracy vs. value plot...")
    
plt.figure()
fig = plt.gcf()
fig.set_size_inches(8, 6)

plt.plot(acc[0,:], acc[1,:], label='Train Accuracy')
plt.plot(acc[0,:], acc[2,:], label='Test Accuracy')
plt.plot(acc[0,:], acc[3,:], label='Validation Accuracy')
plt.xlabel('No. of Nodes')
plt.ylabel('Accuracy (%)')

plt.legend()
plt.show()

fig_name = "1a_one_hot.png"
fig.savefig(fig_name, dpi=100)
    
print(f"Plot saved as {fig_name}...")


# In[41]:


print("Train Accuracy:", acc[1,-1])
print("Validation Accuracy:", acc[3,-1])
print("Test Accuracy:", acc[2,-1])




