# In this code, we generate the data locally so that we may speed up the processing of data in the learning process. 

import argparse
import tensorflow as tf
from dataGenNew import rand_En, rand_costheta, rand_phi, full_measure, compute_waveplate, norm_unitary 
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

shift = 0

# Load configuration file

stream = open(f"configs/datagen.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Define some initial parameters for the generation process

total = eval(cnfg['total'])
res = cnfg['num_pixs']
noise = cnfg['noise']
stateNoise = cnfg['stateNoise']
maxAng = math.radians(cnfg['maxAng'])
isWaveplate = cnfg['isWaveplate']
max_waveplates = cnfg['max_waveplate']
sixMeasure = cnfg['sixMeasure']

filename = cnfg['filename']
applyInverse = cnfg['apply_inversion']

n_coeff_low = cnfg['n_coeffs_low']
n_coeff_high = cnfg['n_coeffs_high']

'''
# Now to generate examples. Let's update this so that there is no ambiguity between train and test dataset examples'
'''

if (sixMeasure):
    X = np.empty((int(total), res, res, 6))
else:
    X = np.empty((int(total), res, res, 5))
    
y = np.empty((int(total), res, res, 3))

for ii in range(int(total)):
    num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
    
    if(isWaveplate):
        a1, a2, a3 = compute_waveplate(num_waveplates, np.random.randint(low=n_coeff_low, high=n_coeff_high),np.random.randint(low=n_coeff_low, high=cnfg['n_coeffs_high']), res, maxAng)
        
        if a2[0,0] > np.pi/2:
            a2 = np.pi - a2
            if (applyInverse):
                a1 = np.pi - a1
                a3 = 2*np.pi - a3
        
        if (a2[0,0] < (np.pi/2 + noise)  and a2[0,0] > (np.pi/2 - noise)) and a3[0,0] > np.pi:
            a3 = 2*np.pi - a3
        
    else: 

        a1=rand_En(np.random.randint(low=n_coeff_low, high=n_coeff_high),np.random.randint(low=n_coeff_low, high=n_coeff_high),res, maxAng) 
        
        # Let's define a list which stores # of coefficents featured by each sample 
        
        freqs_nx = np.array([np.random.randint(low=n_coeff_low, high=n_coeff_high), np.random.randint(low=n_coeff_low, high=n_coeff_high)]) # Freq_nx[0] refers to # of coeffs in x direction. freq_ny[0] refers to # of coeffs in y direction
        freqs_ny = np.array([np.random.randint(low=n_coeff_low, high=n_coeff_high), np.random.randint(low=n_coeff_low, high=n_coeff_high)]) 
        freqs_nz = np.array([np.random.randint(low=n_coeff_low, high=n_coeff_high), np.random.randint(low=n_coeff_low, high=n_coeff_high)]) 
       
        # Compute the sum of each frequency list 
        
        sum_nx = np.sum(freqs_nx)
        sum_ny = np.sum(freqs_ny)
        sum_nz = np.sum(freqs_nz)
        
        # Obtain cartesian coordinates
        
        nx = rand_nx(freqs_nx[0], freqs_nx[1], res, maxAng)
        ny = rand_ny(freqs_ny[0], freqs_ny[1], res, maxAng)
        nz = rand_nz(freqs_nz[0], freqs_nz[1], res, maxAng)
        
        # First, normalize cartesian coordinates 
        
        nx,ny,nz = norm_unitary(nx, ny, nz, sum_nx, sum_ny, sum_nz)
        
        # Perform the inversion of the process. This is either a partial inversion on nz, or a full inversion of the process depending on applyInverse
        
        if nz[0,0] < 0:
            nz = -nz
            if(applyInverse):
                a1 = (np.pi - a1)
                nx = -nx
                ny = -ny
                
        # If nz is not small (depending on the noise), or if nx is positive, then we invert nx
        if (nz[0,0] < noise and nz[0,0] > -noise) and nx[0,0] < 0:
            nx = -nx 
            
        # Now, convert to spherical coordinates
        a2 = np.arccos(nz)
        a3 = np.arctan2(ny, nx) 
    
        # rewrap phi
        for i in range(res):
            for j in range(res):
                if a3[i,j] < 0:
                    a3[i,j] += 2*np.pi
        
    y[ii,:,:,0] = a1
    y[ii,:,:,1] = a2
    y[ii,:,:,2] = a3
        
    X[ii] = full_measure(a1,a2,a3,res,noise, stateNoise, sixMeasure=sixMeasure)
    
    if ii % 50 == 0:
        print('Data generated: ', ii, '/', total)
        

# Dump everything in one pickle file

with open(filename + '.pkl', 'wb') as f:
    pickle.dump([X, y], f)
    

    

    

             