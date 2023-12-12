# In this code, we generate the data locally so that we may speed up the processing of data in the learning process. 

import argparse
import tensorflow as tf
from dataGenNew import rand_En, rand_costheta, rand_phi, full_measure, compute_waveplate
from dataGenNew import rand_nx, rand_ny, rand_nz
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pickle
import yaml
from yaml import Loader
import random
import math


# parse through slurm array (for use w/ bash script. You can set shift to be a random integer for the purposes of testing locally)

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

# Load configuration file
stream = open(f"configs/datagen{shift}.yaml", 'r')
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
applyInverse = cnfg['apply_inversion']

'''
# Now to generate training examples
'''

X_train = np.empty((int(train), res, res, 5))
y_train = np.empty((int(train), res, res, 3))

for ii in range(int(normal_train)+int(special_train)):
    num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
    fac = np.random.uniform(0, 0.00001)
    if(isWaveplate):
        a1, a2, a3 = compute_waveplate(num_waveplates, np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']),np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
    else:
        a1=rand_En(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']),np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']),res, maxAng) 
        #obtain cartesian coordinates
        nx = rand_nx(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
        ny = rand_ny(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
        nz = rand_nz(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)

        # First, normalize cartesian coordinates 
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx = nx/norm 
        ny = ny/norm 
        nz = nz/norm 
        
        nz = np.zeros((res, res)) + 0.01

        # Perform the inversion
        
        if nz[0,0] < 0:
            nz = -nz
            if(applyInverse):
                a1 = (np.pi - a1)
                nx = -nx
                ny = -ny
                
        # If nz is not small (depending on the noise), or if nx is positive, then we break out of the loop
        
        if (nz[0,0] < noise and nz[0,0] > -noise) and nx[0,0] < 0:
            nx = -nx 

        # Now, convert to spherical coordinates
        
        if (ii > normal_train):
                a2=fac*np.arccos(nz)
        else:
                a2=np.arccos(nz)
        
        a3 = np.arctan2(ny, nx)  
    
    # In the special case where a2 is very near pi/2, perform another inversion s.t. process is confined to one quarter of the bloch sphere
    
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
    fac = np.random.uniform(0,0.001)
    num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
    
    if(isWaveplate):
        a1, a2, a3 = compute_waveplate(num_waveplates, np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
    else:
        
        a1=rand_En(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']),np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']),res, maxAng) 
        #obtain cartesian coordinates
        nx = rand_nx(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
        ny = rand_ny(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)
        nz = rand_nz(np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), np.random.randint(low=cnfg['n_coeffs_low'], high=cnfg['n_coeffs_high']), res, maxAng)

        # First, normalize cartesian coordinates 
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx = nx/norm 
        ny = ny/norm 
        nz = nz/norm 

        # Perform the inversion
        
        if nz[0,0] < 0:
            nz = -nz
            if(applyInverse):
                a1 = (np.pi - a1)
                nx = -nx
                ny = -ny
                
        # If nz is not small (depending on the noise), or if nx is positive, then we break out of the loop
        
        if (nz[0,0] < noise and nz[0,0] > -noise) and nx[0,0] < 0:
            nx = -nx 
            
        #convert to spherical coordinates
        
        if(ii > normal_test):
                a2=fac*np.arccos(nz)
        else:
                a2=np.arccos(nz)
                
        a3 = np.arctan2(ny, nx)
        
        for i in range(res):
            for j in range(res):
                if a3[i,j] < 0:
                    a3[i,j] += 2*np.pi

    y_test[ii,:,:,0] = a1
    y_test[ii,:,:,1] = a2
    y_test[ii,:,:,2] = a3
        
    X_test[ii] = full_measure(a1,a2,a3,res,noise, stateNoise)
    
    if ii % 50 == 0:
        print('Training data: ', ii, '/',normal_train)

# Dump everything (test examples)

with open(filename_test + '.pkl', 'wb') as f:
    pickle.dump([X_test, y_test], f)
    
    

             