#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas
import numpy
import matplotlib.pyplot as plt
import time


# In[1]:

print("Defining class neural_network...")


class neural_network:
    
    # implements a fully connected neural network architecture
    
    def __init__(self, M, n, hidden_layer_architecture, num_targets, learning_rate, adaptive, use_relu):
        
        self.M = M # mini-batch size
        self.n = n # no. of attributes
        self.hidden_layer_architecture = hidden_layer_architecture # list containing no. of perceptrons in corresponding hidden layer
        self.r = num_targets # no. of target classes
        self.learning_rate = learning_rate
        self.adaptive = adaptive
        self.use_relu = use_relu
        
        self.parameters = {}
        input_dim = self.n + 1
        for i in range(len(hidden_layer_architecture)):
            layer_size = hidden_layer_architecture[i]
            self.parameters["w"+str(i)] = numpy.random.randn(input_dim, layer_size) * numpy.sqrt(2/input_dim)
            self.parameters["w"+str(i)][0,:] = 0
            # print(self.parameters["w"+str(i)])
            input_dim = layer_size
        self.parameters["w"+str(len(hidden_layer_architecture))] = numpy.random.randn(input_dim, self.r) * numpy.sqrt(2/input_dim) # output layer
        # print(self.parameters["w"+str(i)])
        
        self.output = None # current mini-batch outputs        
        self.all_inputs = {} # current mini-batch hidden layer inputs
        self.back_deltas = {} # current mini-batch deltas, delta_layer_i [0, j] = del(J(theta))/del(net_j)    
        
    def one_hot_encode_target(self, subset):
        enc = numpy.zeros((subset.size, self.r))
        enc[numpy.arange(subset.size), subset] = 1
        return enc
            
    def sigmoid(self, u):
        return 1.0/(1+numpy.exp(-1*u))
    
    def relu(self, u):
        return numpy.maximum(0, u)
    
    def forward_propagation(self, b):
        # b is mini_batch number 
        
        self.all_inputs = {}        
        x_prev = self.train_X[(b-1)*self.M: b*self.M, :]
        self.all_inputs["x"+str(0)] = x_prev
        
        if self.use_relu:
            
            for i in range(len(self.hidden_layer_architecture)):
                parameters = self.parameters["w"+str(i)]
                X_theta = numpy.matmul(x_prev, parameters) # Z is (m, prev_hl_dim)*(prev_hl_dim, hl_dim)
                Z = self.relu(X_theta)
                x_prev = Z
                self.all_inputs["x"+str(i+1)] = x_prev
            
        else: # sigmoid     

            for i in range(len(self.hidden_layer_architecture)):
                parameters = self.parameters["w"+str(i)]
                X_theta = numpy.matmul(x_prev, parameters) # Z is (m, prev_hl_dim)*(prev_hl_dim, hl_dim)
                Z = self.sigmoid(X_theta)
                x_prev = Z
                self.all_inputs["x"+str(i+1)] = x_prev
            
        # output layer
        parameters = self.parameters["w"+str(len(self.hidden_layer_architecture))]
        X_theta = numpy.matmul(x_prev, parameters) # Z is (m, prev_hl_dim)*(prev_hl_dim, r)
        self.output = self.sigmoid(X_theta)
        
    def compute_loss(self, b): 
        # b is mini_batch number
        y = self.one_hot_encode_target(self.train_Y[(b-1)*self.M: b*self.M])
        J = (1/2*self.M) * numpy.sum((y - self.output)*(y - self.output))
        return J
    
    def compute_deltas(self, b):
        # b is mini_batch number
        self.back_deltas = {}
        
        y = self.one_hot_encode_target(self.train_Y[(b-1)*self.M: b*self.M])
        assert y.shape == self.output.shape
        delta_prev = (1.0/self.M)*numpy.sum((y-self.output)*self.output*(1-self.output), axis=0, keepdims=True)     
        self.back_deltas["delta"+str(len(self.hidden_layer_architecture))] = delta_prev
        
        if self.use_relu:
            
            for i in range(len(self.hidden_layer_architecture)-1, -1, -1):
                output_i = self.all_inputs["x"+str(i+1)] 
                assert output_i.shape == (self.M, self.hidden_layer_architecture[i])
                theta = self.parameters["w"+str(i+1)]
                relu_deriv = numpy.where(output_i > 0, 1.0, 0.0) 
                delta = (1.0/self.M)*numpy.sum(relu_deriv*numpy.matmul(delta_prev, theta.T), axis=0, keepdims=True)
                self.back_deltas["delta"+str(i)] = delta
                delta_prev = delta       
        
        else: # sigmoid
        
            for i in range(len(self.hidden_layer_architecture)-1, -1, -1):
                output_i = self.all_inputs["x"+str(i+1)] 
                assert output_i.shape == (self.M, self.hidden_layer_architecture[i])
                theta = self.parameters["w"+str(i+1)]
                delta = (1.0/self.M)*numpy.sum(output_i*(1-output_i)*numpy.matmul(delta_prev, theta.T), axis=0, keepdims=True)
                self.back_deltas["delta"+str(i)] = delta
                delta_prev = delta       
        
    def backward_propagation(self, b, eta):
        # b is mini_batch number
        # eta is learning rate
        self.compute_deltas(b)
        
        for i in range(len(self.hidden_layer_architecture)+1):
            broadcasted_delta = numpy.broadcast_to(self.back_deltas["delta"+str(i)], (self.M, self.back_deltas["delta"+str(i)].shape[1]))
            self.parameters["w"+str(i)] += eta*(1.0/self.M)*numpy.matmul(self.all_inputs["x"+str(i)].T, broadcasted_delta) 
            # (m, prev_hl_dim).T * (1, hl_dim)
            
    def converged(self, loss_values, iteration, num_values):
        if loss_values.size > num_values:
            print(f"\tIteration {iteration}, loss difference: {abs(numpy.mean(loss_values[-1*num_values:]) - numpy.mean(loss_values[-2*num_values:-1*num_values]))}")
            if abs(numpy.mean(loss_values[-1*num_values:]) - numpy.mean(loss_values[-2*num_values:-1*num_values])) < 1e-3:
                return True
        return False
        
    def train(self, dataframe):
        loss_values = numpy.empty((0,))
        iteration = 0
        batch_no = 1
        epoch_no = 1
        eta = self.learning_rate
        
        self.dataframe = dataframe
        
        self.train_X = dataframe.iloc[:, 0:-1].to_numpy()
        intercept = numpy.ones((self.train_X.shape[0], 1))
        self.train_X = numpy.append(intercept, self.train_X, axis=1)
        self.train_Y = dataframe.iloc[:, -1].to_numpy()
        # shuffle the dataset
        perm = numpy.random.permutation(self.train_X.shape[0])
        self.train_X = self.train_X[perm]
        self.train_Y = self.train_Y[perm]
        
        self.total_b = self.train_X.shape[0]//self.M
        
        while True:
            if iteration%250 == 0:
                if self.converged(loss_values, iteration, 250):
                    break
            if self.adaptive:
                eta = self.learning_rate/numpy.sqrt(epoch_no)
            self.forward_propagation(batch_no)
            self.backward_propagation(batch_no, eta)
            loss = self.compute_loss(batch_no)
            loss_values = numpy.append(loss_values, loss)
            iteration += 1
            if batch_no == self.total_b:
                epoch_no += 1
            batch_no = (batch_no)%self.total_b + 1
    
    def predict(self, X):
        x_prev = X        
        for i in range(len(self.hidden_layer_architecture)+1):
            parameters = self.parameters["w"+str(i)]
            X_theta = numpy.matmul(x_prev, parameters) # Z is (m, prev_hl_dim)*(prev_hl_dim, hl_dim)
            Z = self.sigmoid(X_theta)
            x_prev = Z
        prediction = numpy.argmax(x_prev, axis=1)
        return prediction
    
    def compute_accuracy(self, X, y):    
        x_prev = X
        y = y.astype(int)
        for i in range(len(self.hidden_layer_architecture)+1):
            parameters = self.parameters["w"+str(i)]
            # print(x_prev.shape, parameters.shape)
            X_theta = numpy.matmul(x_prev, parameters) # Z is (m, prev_hl_dim)*(prev_hl_dim, hl_dim)
            Z = self.sigmoid(X_theta)
            x_prev = Z
        # print(x_prev[:1000,:])
        prediction = numpy.argmax(x_prev, axis=1)
        # print(prediction.shape)
        # print(prediction[:10000])
        # print(y[:10000])
        return 100*numpy.count_nonzero(prediction==y)/y.size
        
# In[ ]:

print("Defining function for confusion matrix...")


def confusion_matrix(num_classes, y, predicted_y):
    confusion_matrix = numpy.empty((num_classes, num_classes), dtype=numpy.int64)
    for i in range(num_classes):
        label_i_mask = (y==i)
        for j in range(num_classes):
            predicted_j_actual_i = (predicted_y[label_i_mask]==j)        
            # confusion_matrix[i,j] = no. of examples of actual class i predicted as class j
            confusion_matrix[i,j] = numpy.count_nonzero(predicted_j_actual_i)        
    return confusion_matrix

