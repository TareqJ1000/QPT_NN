'''
datagenLocal

Computes the fixed dataset for the training of our networks
'''

import argparse
import tensorflow as tf
import keras

from synthetic_utils import rand_En, full_measure, compute_waveplate, norm_unitary, rand_nx, rand_ny, rand_nz

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pickle
import yaml
from yaml import Loader
import random
import math

# Continuous/fixed data generating classes. On runtime, sequence classes throw away the generated data from memory, making them quite memory efficient for our purposes. 

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, noise=0.01,stateNoise=0.01, n_coeff=10, res=128, batch_size=100, batches_per_epoch=100, alpha=0.3, n_coeff_low=1, n_coeff_high=15, maxAng=math.radians(10), isWaveplates = True, num_waveplates=10, num_waveplates_min=1, num_waveplates_max=2, unitaryParam = 3, isSingle=False, applyInverse = False, sixMeasure = False): #default values
        'Initialization'
        self.batch_size=batch_size # number of datasets per batch size
        self.res=res # resolution to use
        self.n_coeff_high=n_coeff_high # highest number of fourier coeffs considered
        self.n_coeff_low=n_coeff_low # lowest number of fourier coeffs considered
        self.batches_per_epoch=batches_per_epoch # number of batches per epoch
        self.noise=noise # maximum gaussian noise introduced to measurements
        self.stateNoise=stateNoise # by how much do we perturb our polametric measurements?
        self.alpha=alpha # proportion of data that are "normal" and fixed specifically near the pole 
        self.isWaveplates=isWaveplates # do we generate waveplates, or continuous processes?
        self.num_waveplates_min=num_waveplates_min # how many waveplates should we cascade?
        self.num_waveplates_max=num_waveplates_max
        self.maxAng = maxAng # maximum rotation that we apply to the unitary. 
        self.isSingle = isSingle # Do we consider training with only a single output? 
        self.unitaryParam = unitaryParam # for single output training
        self.applyInverse = applyInverse
        self.sixMeasure = sixMeasure # Do we predict w/ six polametric measurements?
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        if (self.isWaveplates and self.isSingle):
            X, y = self.__generate_waveplate_single()
        elif (self.isWaveplates):
            X, y = self.__generate_waveplates()
        else:
            X, y = self.__generate_continuous()

        return X, y
    

    def __generate_waveplates(self):
        'Generates data containing batch_size samples. Here, we generate processes derived from optical waveplates'
        
        # Initialization
        if(self.sixMeasure):
            X = np.empty((self.batch_size,self.res,self.res,6))
        else:
            X = np.empty((self.batch_size,self.res,self.res,5))
            
        y = np.empty((self.batch_size,self.res,self.res,3))
    
        for i in range(0, self.batch_size):
            n_coeff_x = int(np.random.uniform(low=self.n_coeff_low, high=self.n_coeff_high+1)) # Values obtained are from n_coeff_low to n_coeff_high
            n_coeff_y = int(np.random.uniform(low=self.n_coeff_low, high=self.n_coeff_high+1)) # Values obtained are from n_coeff_low to n_coeff_high
            
            num_waveplates = np.random.randint(self.num_waveplates_min, high=self.num_waveplates_max+1)
            a1,a2,a3 = compute_waveplate(num_waveplates,n_coeff_x, n_coeff_y,self.res, self.maxAng) # En, theta, phi
            
            '**inversion step**' 
            
            if a2[0,0]>np.pi/2: # make sure the first pixel has nz>0
                # Essentially, we shift the gauge from one representation to another
                a2=(np.pi-a2)  # theta

            y[i,:,:,0]=a1
            y[i,:,:,1]=a2
            y[i,:,:,2]=a3
            
            X[i]=full_measure(a1,a2,a3,self.res,self.noise, self.stateNoise)
            
        return X, y
    
    
    def __generate_continuous(self):
        'Generates data containing batch_size samples. Here, we generate processes derived from continuous processes'
        # Initialization
        X = np.empty((self.batch_size,self.res,self.res,5))
        y = np.empty((self.batch_size,self.res,self.res,3))
        special_train = int((self.alpha*(self.batch_size)))
        normal_train = self.batch_size - special_train
        fac = 1
        for ii in range(0, normal_train+special_train):
            a1=rand_En(np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high), np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high),self.res, self.maxAng) 
            
            #obtain cartesian coordinates
            
            nx = rand_nx(np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high), np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high),self.res, self.maxAng) 
            ny = rand_ny(np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high), np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high),self.res, self.maxAng) 
            nz = rand_nz(np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high), np.random.randint(low=self.n_coeff_low, high=self.n_coeff_high),self.res, self.maxAng) 
        
            # First, normalize cartesian coordinates 
            
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            nx = nx/norm 
            ny = ny/norm 
            nz = nz/norm 
            
            # Perform the inversion over cartesian coordinates 
            
            if nz[0,0] < 0:
                nz = -nz
                if(self.applyInverse):
                    a1 = (np.pi - a1)
                    nx = -nx
                    ny = -ny
                    
            # If nz is not small (depending on the noise), or if nx is positive, then we break out of the loop
            
            if (nz[0,0] < self.noise and nz[0,0] > -self.noise) and nx[0,0] < 0:
                nx = -nx 
                
            # convert to spherical coordinates
            
            if(ii > normal_train):
                    a2=fac*np.arccos(nz)
            else:
                    a2=np.arccos(nz)
            a3 = np.arctan2(ny, nx)  
            for i in range(self.res):
                for j in range(self.res):
                    if a3[i,j] < 0:
                        a3[i,j] += 2*np.pi
            
            #a3=rand_phi(n_coeff,res,maxAng) 
            
            if a2[0,0]>np.pi/2: # make sure the first pixel has nz>0
                a2=(np.pi-a2)
                
            y[ii,:,:,0] = a1
            y[ii,:,:,1] = a2
            y[ii,:,:,2] = a3
            
            X[ii] = full_measure(a1,a2,a3,self.res,self.noise, self.stateNoise)
            
        return X,y
            

class FixedDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, outputs, batch_size=32, shuffle=True, sigma = 0.01): 
        'Initialization'
        self.inputs = inputs # These are our inputs
        self.outputs = outputs # These are our outputs
        self.number_of_samples = len(self.inputs)
        self.batch_size = batch_size # Size of batches
        self.batches_per_epoch = self.number_of_samples//batch_size # Number of batches per epoch. Note that // automatically implements the floor function. 
        self.shuffle = shuffle # Do we shuffle the dataset? 
        self.sigma = sigma # This controls the random noise that we choose to apply onto the datset in real time
        self.on_epoch_end() 
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        indices = np.arange(self.number_of_samples)
        if self.shuffle == True:
           np.random.shuffle(indices)
           self.inputs = self.inputs[indices]
           self.outputs = self.outputs[indices]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch


    def __getitem__(self, index):
        'Generate one batch of data'
        X = self.inputs[index*self.batch_size:(index+1)*self.batch_size]
        y = self.outputs[index*self.batch_size:(index+1)*self.batch_size]
        'New: we apply transformations onto the images'
        # Apply noise onto our measurements. 
        noiseMaker = np.random.normal(0, self.sigma, np.shape(X))
        X += noiseMaker 
        
        return X, y


if __name__ == '__main__':
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
    

    

    

             