#!/usr/bin/env python
# coding: utf-8

# In[1]:
#all the functions used for data generation are in this script.
#each time it generates (flattened) 2Nx2N matrices representing a NxN 2D map of SU(2) processes, called pixels. The input data will be the measurements (5) for each pixel, the output the parameters E (quasi-energy), nx and ny (two components of the Bloch vector) for each. 
#Given that the stokes measurements are the same for U and -U, we restrict the first pixel to have nz=+sqrt(1-nx^2 -ny^2)>0 (to have a single solution). This local phase choice sets the phase for the entire map.

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import math

import matplotlib.pyplot as plt

# Define Pauli matrices

s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])

# State definitions

l=np.array([1,0])
r=np.array([0,1])

h=np.array([1,1])/np.sqrt(2)
v=np.array([1,-1])/np.sqrt(2)

d=np.array([1,1j])/np.sqrt(2)
a=np.array([1,-1j])/np.sqrt(2)

# Calculates the coefficients of the fourier series
# x - x coordinate map
# y - y coordinate map
# delta_low - lower bound of fourier coeff
# delta_high - higher bound of fourier coeff 
# n_coeff_x - number of fourier coefficents in the x direction 
# n_coeff_y - number of fourier coefficents in the y direction
# ang - angle of rotation of the process. 
# minusOne - If true, then allow negative values in the final map 

def generate_random_function(x, y, delta_low, delta_high, n_coeff_x, n_coeff_y, ang, minusOne = False): #generate random continuous function from a random set of coefficients

    # Dimensions of the coefficent array

    dim_x = n_coeff_x+1
    dim_y = n_coeff_y+1
    
    # Calculate the Fourier series
    
    coefficients_alpha = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_beta = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_delta = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_gamma = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    
    # Compute frequencies in the x and y directions
    
    frequency_x = np.zeros(dim_x)
    frequency_y = np.zeros(dim_y)
    
    for ii in range(dim_x):
        frequency_x[ii] = 2*np.pi*(ii)
        
    for ii in range(dim_y):
        frequency_y[ii] = 2*np.pi*(ii)
    
    # Initialize the function value
    function_value = np.zeros([len(x),len(y)]) #this way it returns a matrix
    
    #random rotation to avoid symmetric output
    x1=np.cos(ang)*x - np.sin(ang) * y
    y1=np.sin(ang)*x + np.cos(ang) * y
    
    # Compute the Fourier series
    for i in range(dim_x):
        for j in range(dim_y):
           
            function_value += coefficients_alpha[i,j]*np.cos(frequency_x[i] * x1) * np.sin(frequency_y[j] * y1) + coefficients_beta[i,j]*np.sin(frequency_x[i] * x1) * np.cos(frequency_y[j] * y1)+ coefficients_delta[i,j]*np.cos(frequency_x[i] * x1) * np.cos(frequency_y[j] * y1)+ coefficients_gamma[i,j]*np.sin(frequency_x[i] * x1) * np.sin(frequency_y[j] * y1)
    
    # Apply normalization to the maps
    
    function_value = function_value/np.max(abs(function_value))
    
    # Constant maps will fall either along +1 or -1, so introduce a random constant
   
    if(n_coeff_x == 0 and n_coeff_y==0):
        function_value = function_value*np.random.uniform(0,1)
    
    # If the function is non-negative, then return (function_value+1)/2
    
    if(minusOne):
        return function_value
    else:
        return (function_value+1)/2
    
'''
Functions for the generation of generic, periodic synthetic processes 
'''

# Random quasi-energy (also called Theta in previous paper), continuous 2d function, discretized on a res x res grid
# n_coeff_x - number of coefficents in the x direction 
# n_coeff_y - number of coefficents in the y direction 
# res - process resolution 
# maxAng - maximum angle of rotation 

def rand_En(n_coeff_x, n_coeff_y, res, maxAng): 
    
    # Randomize the angle
    
    ang=random.uniform(0,maxAng)
    
    # Instantiate the coordinates of the map. The maximum value is shifted by some random value to simulate partial imaging of the plate 
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    # bounds are [0, np.pi]
    
    return np.pi*generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y, ang)
 
# random nx
# n_coeff_x - number of coefficents in the x direction 
# n_coeff_y - number of coefficents in the y direction 
# res - process resolution 
# maxAng - maximum angle of rotation 

def rand_nx(n_coeff_x, n_coeff_y ,res, maxAng): 
    
    # Randomize the angle of rotation
    ang=random.uniform(0,maxAng)
   # Instantiate the coordinates of the map. The maximum value is shifted by some random value to simulate partial imaging of the plate 
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y ,ang, minusOne = True)


# random ny
# n_coeff_x - number of coefficents in the x direction 
# n_coeff_y - number of coefficents in the y direction 
# res - process resolution 
# maxAng - maximum angle of rotation 

def rand_ny(n_coeff_x, n_coeff_y, res, maxAng): #random ny
    # Randomize the angle of rotation
    ang=random.uniform(0,maxAng)
    # Instantiate the coordinates of the map. The maximum value is shifted by some random value to simulate partial imaging of the plate 
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y, ang, minusOne = True)


# random nz
# n_coeff_x - number of coefficents in the x direction 
# n_coeff_y - number of coefficents in the y direction 
# res - process resolution 
# maxAng - maximum angle of rotation 

def rand_nz(n_coeff_x, n_coeff_y, res, maxAng): # rand nz
    # Randomize the angle of rotation
    ang=random.uniform(0,maxAng)
    # Instantiate the coordinates of the map. The maximum value is shifted by some random value to simulate partial imaging of the plate 
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y,-1, 1, n_coeff_x, n_coeff_y, ang, minusOne = True)



        
'''
 FUNCTIONS FOR THE GENERATION OF (CASCADED) WAVEPLATES
'''

# Encodes the pixelwise process matrix of a waveplate
# delta - waveplate retardance
# theta - optic axis modulation 

def waveGen(delta, theta):
    mat = np.zeros([2,2], dtype=complex)
    
    mat[0,0]=np.cos(delta/2)
    mat[0,1] = 1j*np.sin(delta/2)*np.exp(-2j*theta) # theta is bounded up to pi
    mat[1,0] = 1j*np.sin(delta/2)*np.exp(2j*theta) # theta is bounded up to pi
    mat[1,1] = np.cos(delta/2)
    
    return mat

# Generates the optic-axis modulation for the waveplate (ang is in degrees)
# n_coeff_x - number of fourier freqs in the x- direction
# n_coeff_y - number of fourier freqs in the y- direction 
# res - resolution of process 
# maxAng - maximum angle in which the process is rotated

def rand_optic(n_coeff_x, n_coeff_y, res, maxAng):
    ang = random.uniform(0,maxAng)
    
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return np.pi*generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y, ang)
    
# Cascade multiple waveplates together, thereby creating our unitary matrix
# num - number of waveplates to cascade
# n_coeff_x - number of fourier freqs in the x- direction
# n_coeff_y - number of fourier freqs in the y- direction 
# res - resolution of process 
# maxAng - maximum angle in which the process is rotated 

def casOptics(num, n_coeff_x, n_coeff_y, res, maxAng):
    
    unitary = np.identity(2, dtype=complex)
    
    theta_list = []
    delta_list = []
    
    # First, create array of thetas and deltas
    
    for i in range(num):
        theta = rand_optic(n_coeff_x, n_coeff_y, res, maxAng)
        delta = random.uniform(0, 2*np.pi)
        
        theta_list.append(theta)
        delta_list.append(delta)
    
    
    return theta_list, delta_list

# Complete function which generates the unitary for one or more waveplates and returns En, theta, phi

def compute_waveplate(num_waveplate, n_coeff_x, n_coeff_y, res, maxAng):
    
    En = np.zeros([res, res])
    thetapol = np.zeros([res, res])
    phi = np.zeros([res, res])
    
    theta_list, delta_list = casOptics(num_waveplate, n_coeff_x, n_coeff_y, res, maxAng)
    
    for ind1 in range(res):
        for ind2 in range(res):
            # build up the waveplate
            uSyn = np.identity(2, dtype=complex)
            
            for i in range(num_waveplate):
                uSyn = np.dot(uSyn, waveGen(delta_list[i], theta_list[i][ind1][ind2]))
            
            En_pix = retrieve_En(uSyn)
            #polar coordinates
            thetapol_pix = retrieve_thetapol(uSyn, En_pix)
            
            phi_pix = retrieve_phi(uSyn, En_pix)
            
            if (phi_pix<0):
                phi_pix += 2*np.pi
                
            En[ind1,ind2] = En_pix
            thetapol[ind1,ind2] = thetapol_pix
            phi[ind1,ind2] = phi_pix
    
    return En, thetapol, phi




'''
 FUNCTIONS TO RETRIEVE UNITARY PARAMETERS
'''
# Retrieves En (or big THETA)
# unitary - unitary process matrix

def retrieve_En(unitary):
    return np.arccos(0.5*np.trace(unitary))

# Retrieves nx (En must be computed for that unitary) 
# unitary - unitary process matrix
# En - Big theta unitary parameter

def retrieve_nx(unitary, En):
    return (1j/(2*np.sin(En)))*np.trace(unitary*s1)

# Retrieves ny (En must be computed for that unitary) 
# unitary - unitary process matrix
# En - Big theta unitary parameter

def retrieve_ny(unitary, En):
    return (1j/(2*np.sin(En)))*np.trace(unitary*s2)

# Retrieves nz (En must be computed for that unitary) 


def retrieve_nz(unitary, En):
    return (1j/(2*np.sin(En)))*np.trace(unitary*s3)

# Retrives theta and phi for unitary (to map in the sphere)

def retrieve_thetapol(unitary, En):
    return np.arccos(retrieve_nz(unitary, En))

def retrieve_phi(unitary,En):
    nx = np.real(retrieve_nx(unitary, En))
    ny = np.real(retrieve_ny(unitary, En))
    return np.arctan2(ny,nx)

# This is a special normalization function which handles normalization case by case
# nx, ny, nz - unitary params in cartesian coordinates
# sum_nx, sum_ny, sum_nz - sum of fourier frequencies in x- and y- directions 

def norm_unitary(nx,ny,nz,sum_nx,sum_ny,sum_nz):
    
    print(f"nx: {sum_nx}, ny: {sum_ny}, nz: {sum_nz}")
   
    res = np.shape(nx)[0]
    nx_new = np.zeros((res,res))
    ny_new = np.zeros((res,res))
    nz_new = np.zeros((res,res))
    
    # To handle processes w/ two constant parameters, we filp a coin
    
    roulette = np.random.randint(0,2)
    print(f"coin: {roulette}")
    
    if (sum_nx == 0 and sum_ny == 0 and sum_nz == 0) or (sum_nx > 0 and sum_ny > 0 and sum_nz > 0): # Normalize normally
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        
        nx_new = nx/norm
        ny_new = ny/norm
        nz_new = nz/norm 
        
        print('1')
    
    elif (sum_nx==0 and sum_ny > 0 and sum_nz > 0) or (sum_nx==0 and sum_ny==0 and roulette==0) or (sum_nx==0 and sum_nz==0 and roulette==0) : # This means that nx is a constant process
        norm = np.sqrt((ny**2 + nz**2)/(1-nx**2))
        
        # leave nx as is, but renorm the others. If we have two constant processes, nx is fixed, while ny or nz becomes variable due to the norming. 
        nx_new = nx
        ny_new = ny/norm
        nz_new = nz/norm 
        
        
        print('2')
        
    elif (sum_ny==0 and sum_nx > 0 and sum_nz > 0) or (sum_nx==0 and sum_ny==0 and roulette==1) or (sum_ny==0 and sum_nz==0 and roulette==0): # Now ny is a constant process here
        norm = np.sqrt((nx**2 + nz**2)/(1-ny**2))
        
        # Leave ny as is, but renorm the others
        nx_new = nx/norm 
        ny_new = ny
        nz_new = nz/norm 
        
        print('3')
    
    elif (sum_nz==0 and sum_nx > 0 and sum_ny > 0) or (sum_nx==0 and sum_nz==0 and roulette==1) or (sum_ny==0 and sum_nz==0 and roulette==1): # Now nz is a constant process here 
        norm = np.sqrt((ny**2 + nx**2)/(1-nz**2))
        
        # Leave nz as is, but renorm the others 
        nx_new = nx/norm
        ny_new = ny/norm
        nz_new = nz
        
        print('4')
        
    
    return nx_new, ny_new, nz_new

# Unitary through polar-coordinate parameters. This function is applied on a single pixel, but there is a faster way of doing this
# En, th, phi - the spherical unitary parameters
 
def Ugen(En,th,phi):

    mat=np.zeros([2,2],dtype=complex)
    
    nx=np.sin(th)*np.cos(phi)
    ny=np.sin(th)*np.sin(phi)
    nz=np.cos(th)
    
    mat[0,0]=np.cos(En) - 1j*np.sin(En)*nz
    mat[0,1]=-1j*np.sin(En)*(nx - 1j*ny)
    mat[1,0]=-1j*np.sin(En)*(nx + 1j*ny)
    mat[1,1]=np.cos(En) + 1j*np.sin(En)*nz
    
    return mat

# polarimetric measurement with noise
# in_pol - input polarization
# out_pol - output polarization  
# evo_op - unitary process matrix
# noise - sigma parameter specifying the pixlewise gaussian noise

def measure(in_pol,out_pol,evo_op,noise): # polarimetric measurement with noise
    
    action=np.dot(evo_op,in_pol)
    
    mel=np.vdot(out_pol,action)
    
    pixNoise = random.gauss(0,noise)
    
    result = 0
    
    if(abs(mel)**2 + pixNoise>1) or (abs(mel)**2 + pixNoise < 0):
        result = abs(mel)**2 - pixNoise 
    else:
        result = abs(mel)**2 + pixNoise 
    
    return result

# This applies noise to the states themselves (e.g. if what we project to is not exactly the polarization we want)
# state - state to be perturbed
# randNoise - upper bound for noise to be applied onto state

def perturbState(state, randNoise):
    perturbingState = np.array([random.uniform(0, randNoise), random.uniform(0,randNoise)])
    newState = state + perturbingState
    # Enforce normalization 
    normConst = np.sqrt(np.sum(np.abs(newState)**2))
    return newState/normConst


# Performs the full set of pixelwise measurements on the unitary process. 
# En, th, phi -- the spherical coordinate parameters 
# noise - sigma parameter specifying gaussian pixelwise noise
# stateNoise - upper bound for the uniformly sampled noise applied to the state
# rotateBasis - do we apply a rotation of measurement basis?
# sixMeasure - do we include a sixth measurement? 

def pixel_measure(En,th,phi,noise, stateNoise, rotateBasis = False, sixMeasure = False):
    
    # Let's introduce slight perturbations onto the states that we choose to measure. 
    # The noise needs to be uncorrelated 
    
    unit_op=Ugen(En,th,phi)
    
    if (sixMeasure):
        results=np.zeros([6])
    else: 
        results=np.zeros([5])
      
    results[0]=measure(perturbState(l, stateNoise), perturbState(l, stateNoise), unit_op,noise)
    
    if(rotateBasis):
        results[1]=measure(perturbState(h, stateNoise), perturbState(l, stateNoise), unit_op, noise)
    else:
        results[1]=measure(perturbState(l, stateNoise), perturbState(h, stateNoise), unit_op, noise)
        
    results[2]=measure(perturbState(l, stateNoise), perturbState(d, stateNoise),unit_op,noise)
            
    results[3]=measure(perturbState(h, stateNoise), perturbState(h, stateNoise),unit_op,noise)
    
    results[4]=measure(perturbState(h, stateNoise), perturbState(d, stateNoise),unit_op,noise)
    
    if(sixMeasure):
        results[5]=measure(perturbState(h, stateNoise), perturbState(l, stateNoise), unit_op, noise)
    
    return results

# Performs the full set of measurements on the unitary process over the whole image. 
# En, th, phi -- the spherical coordinate parameters 
# noise - sigma parameter specifying gaussian pixelwise noise
# stateNoise - upper bound for the uniformly sampled noise applied to the state
# rotateBasis - do we apply a rotation of measurement basis?
# sixMeasure - do we include a sixth measurement? 

def full_measure(En,th,phi,res,noise, stateNoise, rotateBasis = False, sixMeasure = False): #measures each pixel and returns flattened res^2 X dim_in vector (dim_in=5)

    if (sixMeasure):
        results=np.zeros([res,res,6])
    else:
        results=np.zeros([res,res,5])

    for ind1 in range(res):
        for ind2 in range(res):
            results[ind1,ind2,:]=pixel_measure(En[ind1,ind2],th[ind1,ind2],phi[ind1,ind2],noise, stateNoise, rotateBasis = rotateBasis, sixMeasure=sixMeasure)
        
    return results

# applies a reordering of measurements, implementing a rotation of basis. Only works for 5 measurements 
# full_meas - array of five measurements
# res - resolution of process matrix. 

def full_measure_reorder(full_meas, res):
    temp_full_meas = np.zeros([res,res,5])
    temp_full_meas[:,:,0] = full_meas[:,:,3]
    temp_full_meas[:,:,1] = full_meas[:,:,1]
    temp_full_meas[:,:,2] = full_meas[:,:,4]
    temp_full_meas[:,:,3] = full_meas[:,:,0]
    temp_full_meas[:,:,4] = full_meas[:,:,2]
    
    return temp_full_meas
    
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
            
            
            
    def __generate_waveplate_single(self):
            'Generates data containing batch_size samples. Here, we generate processes derived from optical waveplates'
            
            # Initialization
            X = np.empty((self.batch_size,self.res,self.res,5))
            y = np.empty((self.batch_size,self.res,self.res))
        
            for i in range(0, self.batch_size):
                n_coeff = int(np.random.uniform(low=self.n_coeff_low, high=self.n_coeff_high+1)) # Values obtained are from n_coeff_low to n_coeff_high
                num_waveplates = np.random.randint(self.num_waveplates_min, high=self.num_waveplates_max+1)
                a1,a2,a3 = compute_waveplate(num_waveplates,n_coeff,self.res, self.maxAng) # En, theta, phi
                
                '**inversion step**' 
                
                if a2[0,0]>np.pi/2: # make sure the first pixel has nz>0
                    # Essentially, we shift the gauge from one representation to another
                    a2=(np.pi-a2)  # theta
                
                # Idea: let's learn each unitary parameter seperately 
                if (self.unitaryParam == 1):
                    a1 = a1/np.max(a1) # renormalization 
                    y[i,:,:]=a1
                
                elif (self.unitaryParam == 2):
                    a2 = a2/np.max(a2) # renormalization
                    y[i,:,:]=a2
                
                else:
                    a3 = a3/np.max(a3) # renormalization
                    y[i,:,:]=a3
                
                X[i]=full_measure(a1,a2,a3,self.res,self.noise, self.stateNoise)
                
            return X, y
            
# In[ ]:
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



