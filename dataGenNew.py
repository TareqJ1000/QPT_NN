#!/usr/bin/env python
# coding: utf-8

# In[1]:
#all the functions used for data generation are in this script.
#each time it generates (flattened) 2Nx2N matrices representing a NxN 2D map of SU(2) processes, called pixels. The input data will be the measurements (5) for each pixel, the output the parameters E (quasi-energy), nx and ny (two components of the Bloch vector) for each. 
#Given that the stokes measurements are the same for U and -U, we restrict the first pixel to have nz=+sqrt(1-nx^2 -ny^2)>0 (to have a single solution). This local phase choice sets the phase for the entire map.

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


def generate_random_function(x, y, coefficients,ang): #generate random continuous function from a random set of coefficients
    
    # Calculate the Fourier series
    n = np.arange(0, len(coefficients))  
    frequency = 2 * np.pi * n    # frequencies
    
    # Initialize the function value
    function_value = np.zeros([len(x),len(y)]) #this way it returns a matrix
    
    #random rotation to avoid symmetric output
    x1=np.cos(ang)*x - np.sin(ang) * y
    y1=np.sin(ang)*x + np.cos(ang) * y
    
    # Compute the Fourier series
    for i in range(len(coefficients)):
        function_value += coefficients[i] * np.cos(frequency[i] * x1) * np.sin(frequency[i] * y1)
      
    return function_value

def norm_rand_f(x, y, coefficients,ang): #make sure the random function is restricted between 0 and 1
    
    random_function = generate_random_function(x, y, coefficients,ang)
    random_function = (np.max(random_function)>1)*random_function/np.max(random_function) + (np.max(random_function)<1)*random_function
    
    return random_function

def rand_En(n_coeff,res): #random quasi-energy (also called Theta in previous paper), continuous 2d function, discretized on a res x res grid
    
    coefficients = np.random.rand(n_coeff)
    ang=random.uniform(0,2*np.pi)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    
    return np.pi*norm_rand_f(x, y, coefficients,ang)

def rand_theta(n_coeff,res): #theta on the Bloch sphere, continuous 2d function, discretized on a res x res grid
    
    coefficients = np.random.rand(n_coeff)
    ang=random.uniform(0,2*np.pi)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    
    return np.pi*norm_rand_f(x, y, coefficients,ang)

def rand_phi(n_coeff,res): #phi on the Bloch sphere, continuous 2d function, discretized on a res x res grid
    
    coefficients = np.random.rand(n_coeff)
    ang=random.uniform(0,2*np.pi)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    
    return 2*np.pi*norm_rand_f(x, y, coefficients,ang)
    

def Ugen(En,th,phi): #unitary from the parameters
    mat=np.zeros([2,2],dtype=complex)
    
    nx=np.sin(th)*np.cos(phi)
    ny=np.sin(th)*np.sin(phi)
    nz=np.cos(th)
    
    mat[0,0]=np.cos(En) - 1j*np.sin(En)*nz
    mat[0,1]=-1j*np.sin(En)*(nx - 1j*ny)
    mat[1,0]=-1j*np.sin(En)*(nx + 1j*ny)
    mat[1,1]=np.cos(En) + 1j*np.sin(En)*nz
    
    return mat


def measure(in_pol,out_pol,evo_op,noise): #polarimetric measurement with noise
    
    action=np.dot(evo_op,in_pol)
    
    mel=np.vdot(out_pol,action)
    
    return abs(abs(mel)**2 + random.gauss(0,noise))

sig_x=np.array([[0,1],[1,0]])
sig_y=np.array([[0,-1j],[1j,0]])
sig_z=np.array([[1,0],[0,-1]])

def pixel_measure(En,th,phi,dim_in,noise):
    
    unit_op=Ugen(En,th,phi)
    
    results=np.zeros([dim_in])
      
    results[0]=measure(l,l,unit_op,noise)
            
    results[1]=measure(l,h,unit_op,noise)
    
    results[2]=measure(l,d,unit_op,noise)
            
    results[3]=measure(h,h,unit_op,noise)
    
    results[4]=measure(h,d,unit_op,noise)
    
    return results

def full_measure(En,th,phi,dim_in,res,noise): #measures each pixel and returns flattened res^2 X dim_in vector (dim_in=5)
    
    results=np.zeros([res,res,dim_in])

    for ind1 in range(res):
        for ind2 in range(res):
            results[ind1,ind2,:]=pixel_measure(En[ind1,ind2],th[ind1,ind2],phi[ind1,ind2],dim_in,noise)
            
        
    return results


# In[3]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, noise=0.01, n_coeff=10, res=128, num_evo=100, batches_per_epoch=100): #default values
        'Initialization'
        self.batch_size = num_evo
        self.num_evo=num_evo
        self.res = res
        self.n_coeff = n_coeff
        self.batches_per_epoch=batches_per_epoch
        self.noise=noise
        
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
        X = np.empty((self.batch_size,self.res,self.res,5))
        y = np.empty((self.batch_size,self.res,self.res,3))
        
        # Generate data
          
        for i in range(self.num_evo):
            
            a1=rand_En(self.n_coeff,self.res)
            a2=rand_theta(self.n_coeff,self.res)
            a3=rand_phi(self.n_coeff,self.res)
            
            if a2[0,0]>np.pi/2: #make sure the first pixel has nz>0
                a2=np.pi-a2
            
            y[i,:,:,0]=a1
            y[i,:,:,1]=a2
            y[i,:,:,2]=a3
            
            X[i]=full_measure(a1,a2,a3,5,self.res,self.noise)
            
            # Display the input to the data 
            
            
        
        return X, y


# In[ ]:




