'''
datagenLocal

Computes the fixed dataset for the training of our networks
'''

import argparse
import tensorflow as tf

from dataGenNew import rand_En, full_measure, compute_waveplate, norm_unitary 
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

#shift = 0

# Load configuration file

stream = open(f"configs/datagen.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Define some initial parameters for the generation process

total = eval(cnfg['total']) # Total number of examples in the dataset
res = cnfg['num_pixs'] # image resolution (in format res x res)
noise = cnfg['noise'] # sigma for noise
stateNoise = cnfg['stateNoise'] # * DEPRECEATED * sigma for state noise
maxAng = math.radians(cnfg['maxAng']) # maximum angle of rotation
isWaveplate = cnfg['isWaveplate'] # Do we generate waveplates
max_waveplates = cnfg['max_waveplate'] # Maximum number of waveplates 
sixMeasure = cnfg['sixMeasure'] # Do we include a sixth measurement?
filename = cnfg['filename'] # Directory name of dataset
# applyInverse = cnfg['apply_inversion'] # Do we apply a harder inversion on the dataset?
n_coeff_low = cnfg['n_coeffs_low'] # lower bound for the number of fourier frequencies in the x- or y- directions
n_coeff_high = cnfg['n_coeffs_high'] # Upper bound for the number of fourier frequencies in the x- or y- directions

# Instantiate the input/output arrays X/y that will hold our dataset. 

if (sixMeasure): # Include 6 measurements
    X = np.empty((int(total), res, res, 6))
else: # Include 5 measurements
    X = np.empty((int(total), res, res, 5))
    
# Holds the unitary parameters in spherical coordinates

y = np.empty((int(total), res, res, 3))

for ii in range(int(total)):
    
    # Case 1: We generate stacks of up to max_waveplates optical waveplates
    if(isWaveplate):
        num_waveplates = np.random.randint(low=1, high=max_waveplates+1)
        
        # Compute the spherical unitary parameters associated with the process. Note that the # of fourier frequencies are randomized in either direction. 
        a1, a2, a3 = compute_waveplate(num_waveplates, np.random.randint(low=n_coeff_low, high=n_coeff_high),np.random.randint(low=n_coeff_low, high=cnfg['n_coeffs_high']), res, maxAng)
        
        # If the first pixel of the process falls on the south pole, apply inversion so that it is on the north pole. 
        if a2[0,0] > np.pi/2:
            a2 = np.pi - a2
        
        # If theta is very close to the equator of the bloch sphere, then we also invert phi. This, in particular, confines nz=0 processes to one quadrant of the northern bloch hemisphere. 
        if (a2[0,0] < (np.pi/2 + noise)  and a2[0,0] > (np.pi/2 - noise)) and a3[0,0] > np.pi:
            a3 = 2*np.pi - a3
        
    else:
        # We generate the unitary params of a random synthetic process from periodic functions
        
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
        
        # normalize cartesian coordinates according to a specific rule depending if the params are constant or not. 
        
        nx,ny,nz = norm_unitary(nx, ny, nz, sum_nx, sum_ny, sum_nz)
        
        # Perform the inversion of the process. Here we perform a false inversion of sorts on processes below the bloch sphere equator, but the net effect is that we want to put everything in the bloch sphere
        if nz[0,0] < 0:
            nz = -nz
                
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
    
    # Regardless of which type of process we choose to gen, save it to the output array 
    
    y[ii,:,:,0] = a1
    y[ii,:,:,1] = a2
    y[ii,:,:,2] = a3
    
    # With the unitary parameters determined, perform synthetic measurements and save to input array 
        
    X[ii] = full_measure(a1,a2,a3,res,noise, stateNoise, sixMeasure=sixMeasure)
    
    if ii % 50 == 0:
        print('Data generated: ', ii, '/', total)
        

# Dump everything in one pickle file

with open(filename + '.pkl', 'wb') as f:
    pickle.dump([X, y], f)
    

    

    

             