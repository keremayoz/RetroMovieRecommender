#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random
from numpy import array
import math

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (np.tanh(x) ** 2))
    return np.tanh(x)

def relu(x, derivative=False):
    if (derivative == True):
        return 1. * (x > 0)
    return x * (x > 0)

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
    
def activation_function(x, derivative=False):
    return relu(x, derivative)


# In[5]:


class NeuralNetwork:
    
    def __init__(self, neuron_list):
        self.layer_count = len(neuron_list)
        self.neuron_list = neuron_list
        #self.biases = [np.random.randn(y, 1) for y in neuron_list[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(neuron_list[:-1], neuron_list[1:])] 
        self.biases = [np.zeros((y, 1), dtype=float) for y in neuron_list[1:]]
        #self.weights = [np.full((y, x),1,dtype=float) for x, y in zip(neuron_list[:-1], neuron_list[1:])] 
        self.activations = [np.zeros((x)) for x in neuron_list]
        self.deltas = [np.zeros((x)) for x in neuron_list]
        self.learning_rate = 0.01
        
    def forward_propogation(self, x):
        self.activations[0] = x      
        for i in range(self.layer_count-1):
            self.activations[i+1] = activation_function(np.dot(self.weights[i], self.activations[i])+self.biases[i])
        return self.activations[-1]
    
    def compute_deltas(self,output_labels):
        # Compute last layers' activations
        for i in range(self.neuron_list[-1]):
            self.deltas[-1][i] = 2*activation_function(self.activations[-1][i],True)*(output_labels[i]-activation_function(self.activations[-1][i]))            
        
        # Compute all deltas in all layers
        # l is layer starting from L-1, ending at 1
        for l in range(self.layer_count-2, 0, -1):
            # i is representing neuron count in the layer l
            for i in range(len(self.activations[l])):
                # j is representing next layer's neurons
                for j in range(len(self.activations[l+1])):
                    self.deltas[l][i] = self.deltas[l+1][j]*self.weights[l][j][i]*activation_function(self.activations[l][i],True)
                 
    def back_propogation(self, output_labels):
        # Compute deltas
        self.compute_deltas(output_labels)
        # Update weights 
        # l is layer starting from L-1, ending at 1
        for l in range(0,self.layer_count-1):
            # i is representing neuron count in the layer l
            for i in range(len(self.activations[l])):
                # j is representing next layer's neurons
                for j in range(len(self.activations[l+1])):
                    self.weights[l][j][i] += self.learning_rate*self.deltas[l+1][j]*self.activations[l][i]

                 
    def train(self, x, y, epoch):
        error = []
       
        for e in range(epoch):
            pass_error = 0
            print("Epoch: " + str(e))
            for i in range(len(x)):
                estimation = self.forward_propogation(x[i])
                print("Est: " + str(np.transpose(estimation)))
                print("Out: " + str(np.transpose(y[i])))
                print("")
                pass_error += np.sum((estimation - y[i])**2) / len(x[i])
                self.back_propogation(y[i])
            error.append(pass_error / len(x))
        return error          


# In[6]:


'''
np.set_printoptions(suppress=True)
neuron_list = [4,2,4]
nn = NeuralNetwork(neuron_list)
# Give input in nx1 dimension
inp = []
for i in range(1000):
    inp.append(np.transpose([abs(np.random.normal(0, 1, 4))]))

err = nn.train(inp,inp, 10)
#plt.plot(err)
#plt.show()
#print(err)
'''


# In[ ]:


'''
neuron_list = [8,4,4,8]
nn = NeuralNetwork(neuron_list)
# Give input in nx1 dimension
inp = np.transpose([[4,5,6,7,8,9,10,11]])
nn.forward_propogation(inp)
out = np.transpose([[4,5,6,7,8,9,10,11]])
nn.compute_deltas(out)
nn.back_propogation(out)
print(nn.deltas)
'''

