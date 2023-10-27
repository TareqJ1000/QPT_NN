# In this code, we generate the data locally so that we may speed up the processing of data in the learning process. 

import argparse
import tensorflow as tf
from dataGenNew import rand_En, rand_costheta, rand_phi, full_measure, compute_waveplate
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pickle
import yaml
from yaml import Loader
import random
import math

# Load yaml file 

stream = open(f"configs/datagen1.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Define some initial parameters for the gneration process

total = eval(cnfg['total'])
res = cnfg['num_pixs']
noise = cnfg['noise']
stateNoise = cnfg['stateNoise']
maxAng = math.radians(cnfg['maxAng'])
isWaveplate = cnfg['isWaveplate']
max_waveplates = cnfg['max_waveplate']

'''
# parameters that control the proportion of training:test case examples\
'''

alpha = cnfg['alpha'] # % of training examples that are special cases
alpha2 = cnfg['alpha2'] # % of test examples that are special cases (default 50%)

test = 0.15*total
train = 0.85*total

special_train = alpha*train 
normal_train = (1-alpha)*train 

special_test = alpha2*test
normal_test = (1-alpha2)*test

filename_train = cnfg['filename_train']
filename_test = cnfg['filename_test']

'''
# Now to generate training examples
'''

X_train = np.empty((int(train), res, res, 5))
y_train = np.empty((int(train), res, res, 3))
    
for ii in range(int(normal_train)+int(special_train)):
    n_coeff= np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high'])
    fac = np.random.uniform(0,0.001)
    num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
    
    if(isWaveplate):
        a1, a2, a3 = compute_waveplate(num_waveplates, n_coeff, res, maxAng)
    else:
        a1=rand_En(n_coeff,res, maxAng)    
        if(ii > normal_train):
                a2=fac*np.arccos(rand_costheta(n_coeff,res, maxAng))
        else:
                a2=np.arccos(rand_costheta(n_coeff, res, maxAng))
        a3=rand_phi(n_coeff,res,maxAng) 
        
        
    if a2[0,0]>np.pi/2: # make sure the first pixel has nz>0
        a2=(np.pi-a2)
        
    y_train[ii,:,:,0] = a1
    y_train[ii,:,:,1] = a2
    y_train[ii,:,:,2] = a3
        
    X_train[ii] = full_measure(a1,a2,a3,res,noise, stateNoise)
    
    if ii % 50 == 0:
        print('Training data: ', ii, '/',normal_train)

# Dump everything (training examples)

with open(filename_train + '.pkl', 'wb') as f:
    pickle.dump([X_train, y_train], f)
    
############################
# Test Example Cases       #
############################ 

X_test = np.empty((int(test), res, res, 5))
y_test = np.empty((int(test), res, res, 3))

# First, generate normal examples
    
for ii in range(int(normal_test)+int(special_test)):
    n_coeff= np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high'])
    fac = np.random.uniform(0,0.001)
    num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
    
    if(isWaveplate):
        a1, a2, a3 = compute_waveplate(num_waveplates, n_coeff, res, maxAng)
    else:
        a1=rand_En(n_coeff,res, maxAng)    
        if(ii > normal_test):
                a2=fac*np.arccos(rand_costheta(n_coeff,res, maxAng))
        else:
                a2=np.arccos(rand_costheta(n_coeff, res, maxAng))
        a3=rand_phi(n_coeff,res,maxAng) 
    
    if a2[0,0]>np.pi/2: # make sure the first pixel has nz>0
        a2=(np.pi-a2)
        
    y_test[ii,:,:,0] = a1
    y_test[ii,:,:,1] = a2
    y_test[ii,:,:,2] = a3
        
    X_test[ii] = full_measure(a1,a2,a3,res,noise, stateNoise)
    
    if ii % 50 == 0:
        print('Training data: ', ii, '/',normal_train)

# Dump everything (test examples)

with open(filename_test + '.pkl', 'wb') as f:
    pickle.dump([X_test, y_test], f)
             
