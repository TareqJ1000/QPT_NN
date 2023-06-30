#!/usr/bin/env python
# coding: utf-8

# In[1]:
#all the functions used for data generation are in this script.
#each time it generates a matrix from SU(2). The input data will be the stokes measurements (6), the output the parameters E 
#(quasi-energy), nx and ny (two components of the Bloch vector). Given that the stokes measurements are the same for U and -U
#only nz=+sqrt(1-nx^2 -ny^2) is allowed (to have a single solution).

import numpy as np
from tensorflow import keras
import random

# In[2]:


l=np.array([1,0])
r=np.array([0,1])

h=np.array([1,1])/np.sqrt(2)
v=np.array([1,-1])/np.sqrt(2)

d=np.array([1,1j])/np.sqrt(2)
a=np.array([1,-1j])/np.sqrt(2)

def Ugen(En,nx,ny): #unitary from the parameters
    mat=np.zeros([2,2],dtype=complex)
    
    nz=np.sqrt(1-nx**2 -ny**2)
    
    mat[0,0]=np.cos(En) - 1j*np.sin(En)*nz
    mat[0,1]=-1j*np.sin(En)*(nx - 1j*ny)
    mat[1,0]=-1j*np.sin(En)*(nx + 1j*ny)
    mat[1,1]=np.cos(En) + 1j*np.sin(En)*nz
    
    return mat


def measure(in_pol,out_pol,evo_op):
    
    action=np.dot(evo_op,in_pol)
    
    mel=np.vdot(out_pol,action)
    
    return abs(abs(mel)**2 + random.gauss(0,0.02))

sig_x=np.array([[0,1],[1,0]])
sig_y=np.array([[0,-1j],[1j,0]])
sig_z=np.array([[1,0],[0,-1]])

def full_measure(En,nx,ny):
    
    dim=6
    unit_op=Ugen(En,nx,ny)
    
    results=np.zeros([dim])
    ind=0
    
    for in_pol1 in [l,h]: 
        for out_pol1 in [l,h,d]: # l->r should be 1-(l->l) if unitary. Also d input depends on the others
            
            results[ind]=measure(in_pol1,out_pol1,unit_op)
            ind+=1
    
    return results


# In[3]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, num_evo=100, batches_per_epoch=100, dim=(6,3)): #default values
        'Initialization'
        self.dim = dim
        self.batch_size = num_evo
        self.num_evo=num_evo
        self.batches_per_epoch=batches_per_epoch
        print(batches_per_epoch)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dim[0])) #empty is like zeros but doesn't set the values. Probably faster!
        y = np.empty((self.batch_size,self.dim[1]))
     
        
        # Generate data
            

        ind=0    
        for i in range(self.num_evo):
            En=random.uniform(0,np.pi) #random parameters are generated here
            nx_val=random.uniform(0,1)
            ny_val=random.uniform(0,1) #the fraction with respect to sqrt(1-nx^2)
            nx=nx_val*2 -1
            ny=(ny_val*2 -1)*np.sqrt(1-nx**2)
            
            X[ind]=full_measure(En,nx,ny)
            y[ind]=np.array([En/(np.pi),nx_val,ny_val])
            
            ind+=1
        
        return X, y


# In[ ]:




