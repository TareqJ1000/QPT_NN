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

# In[2]:
    
# Calculates the coefficients of the fourier series

def generate_random_function(x, y, delta_low, delta_high, n_coeff_x, n_coeff_y, ang, minusOne = False): #generate random continuous function from a random set of coefficients

    # Dimensions of the coefficent array

    dim_x = n_coeff_x+1
    dim_y = n_coeff_y+1
    
    # Calculate the Fourier series
    
    coefficients_alpha = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_beta = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_delta = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    coefficients_gamma = np.random.uniform(low=delta_low, high=delta_high, size=(dim_x,dim_y))
    
    # Calculate the maximum number of fourier coefficents
    
    # Compute frequencies in the x and y directions
    
    frequency_x = np.zeros(dim_x)
    frequency_y = np.zeros(dim_y)
    
    for ii in range(dim_x):
        frequency_x[ii] = 2*np.pi*(ii)
        
    for ii in range(dim_y):
        frequency_y[ii] = 2*np.pi*(ii)
    
    # In principle, this should be the same thing? 
    
    #n = np.arange(0, n_coeff)  
    #frequency = 2 * np.pi * (n+1)    # frequencies
    
    # Initialize the function value
    function_value = np.zeros([len(x),len(y)]) #this way it returns a matrix
    
    #random rotation to avoid symmetric output
    x1=np.cos(ang)*x - np.sin(ang) * y
    y1=np.sin(ang)*x + np.cos(ang) * y
    
    # Compute the Fourier series
    for i in range(dim_x):
        for j in range(dim_y):
           
            function_value += coefficients_alpha[i,j]*np.cos(frequency_x[i] * x1) * np.sin(frequency_y[j] * y1) + coefficients_beta[i,j]*np.sin(frequency_x[i] * x1) * np.cos(frequency_y[j] * y1)+ coefficients_delta[i,j]*np.cos(frequency_x[i] * x1) * np.cos(frequency_y[j] * y1)+ coefficients_gamma[i,j]*np.sin(frequency_x[i] * x1) * np.sin(frequency_y[j] * y1)
            
    function_value = function_value/np.max(abs(function_value))
    
    if(n_coeff_x == 0 and n_coeff_y==0):
        function_value = function_value*np.random.uniform(0,1)
    
    if(minusOne):
        return function_value
    else:
        return (function_value+1)/2


def generate_random_function_poly(x,y,n_coeff,ang):
    
    # generate coefficents over the range (-2, -1) U (1, 2)
    a,b,c,d = -2, -1, 1, 2
    
    prob = np.array([b-a, d-c])
    prob = prob/prob.sum()
    
    coefficients = np.array([np.random.choice([np.random.uniform(a,b), np.random.uniform(c,d)], p=prob) for _ in range(2*n_coeff+1)])
    
    #coefficients = np.random.uniform(low=-2.0,high=-1.0, size=2*n_coeff+1) # This is a 2*coeff + 1 long list
    res = len(x)
    # Initialize the function value
    function_value = np.zeros([res,res]) #this way it returns a matrix
    
    #random rotation to avoid symmetric output
    x1=np.cos(ang)*x - np.sin(ang) * y
    y1=np.sin(ang)*x + np.cos(ang) * y
    
    # Compute the Fourier series
    for i in range(n_coeff+1):
        if (i == 0): # base case
            function_value += coefficients[0]
        else:
            coinfilp = np.random.randint(0,2)
            if coinfilp == 0:
                function_value += coefficients[2*i - 1]*(np.pi/res)*x1**i 
            if coinfilp == 1:
                function_value += coefficients[2*i]*(np.pi/res)*y1**i
            
    #function_value += coefficients[0] + (np.pi*coefficients[1]/res)*x1 + (np.pi*coefficients[2]/res)*y1 + (np.pi*coefficients[3]/res)*x1**2 + (np.pi*coefficients[4]/res)*y1**2 + (np.pi*coefficients[5]/res)*x1**3 + (np.pi*coefficients[6]/res)*y1**3 + (np.pi*coefficients[7]/res)*x1**4 + (np.pi*coefficients[8]/res)*y1**4 + (np.pi*coefficients[9]/res)*x1**5 + (np.pi*coefficients[10]/res)*y1**5 +  (np.pi*coefficients[11]/res)*x1**6 + (np.pi*coefficients[12]/res)*y1**6 +  (np.pi*coefficients[13]/res)*x1**7 + (np.pi*coefficients[14]/res)*y1**7                                                             
    function_value = function_value/np.max(abs(function_value))
    
    return (function_value+1)/2
    
    
def rand_En(n_coeff_x, n_coeff_y, res, maxAng): #random quasi-energy (also called Theta in previous paper), continuous 2d function, discretized on a res x res grid
    
    ang=random.uniform(0,maxAng)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return np.pi*generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y, ang)
 
def rand_costheta(n_coeff_x, n_coeff_y ,res, maxAng): #random quasi-energy (also called Theta in previous paper), continuous 2d function, discretized on a res x res grid
    
    ang=random.uniform(0,2*np.pi)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return 2*(generate_random_function(X, Y, 0, 1, n_coeff_x, n_coeff_y, ang)) - 1

def rand_phi(n_coeff_x, n_coeff_y, res, maxAng): #random quasi-energy (also called Theta in previous paper), continuous 2d function, discretized on a res x res grid
    
    ang=random.uniform(0,maxAng)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0,1)
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return 2*np.pi * (generate_random_function(X, Y, 0, 1, n_coeff_x, n_coeff_y,ang))

def rand_nx(n_coeff_x, n_coeff_y ,res, maxAng): #random nx
    
    ang=random.uniform(0,maxAng)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y, -1, 1, n_coeff_x, n_coeff_y ,ang, minusOne = True)

def rand_ny(n_coeff_x, n_coeff_y, res, maxAng): #random ny
    
    ang=random.uniform(0,maxAng)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y,-1, 1, n_coeff_x, n_coeff_y, ang, minusOne = True)

def rand_nz(n_coeff_x, n_coeff_y, res, maxAng): # rand nz

    ang=random.uniform(0,maxAng)
            
    x_min = random.uniform(0,1)
    x_max = x_min + random.uniform(0.5,1) # How much of the plate do we want to shine?
    y_min = random.uniform(0,1)
    y_max = y_min + random.uniform(0.5,1)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return generate_random_function(X, Y,-1, 1, n_coeff_x, n_coeff_y, ang, minusOne = True)


def Ugen(En,th,phi): #unitary through polar-coordinate parameters
    mat=np.zeros([2,2],dtype=complex)
    
    nx=np.sin(th)*np.cos(phi)
    ny=np.sin(th)*np.sin(phi)
    nz=np.cos(th)
    
    mat[0,0]=np.cos(En) - 1j*np.sin(En)*nz
    mat[0,1]=-1j*np.sin(En)*(nx - 1j*ny)
    mat[1,0]=-1j*np.sin(En)*(nx + 1j*ny)
    mat[1,1]=np.cos(En) + 1j*np.sin(En)*nz
    
    return mat


def Ugen_cart(En, nx, ny, nz): #unitary from cartesian parameters
    mat=np.zeros([2,2],dtype=complex)
    
    mat[0,0]=np.cos(En) - 1j*np.sin(En)*nz
    mat[0,1]=-1j*np.sin(En)*(nx - 1j*ny)
    mat[1,0]=-1j*np.sin(En)*(nx + 1j*ny)
    mat[1,1]=np.cos(En) + 1j*np.sin(En)*nz
    
    return mat


'''
 FUNCTIONS FOR THE GENERATION OF (CASCADED) WAVEPLATES
'''

# Encodes a waveplate

def waveGen(delta, theta):
    mat = np.zeros([2,2], dtype=complex)
    
    mat[0,0]=np.cos(delta/2)
    mat[0,1] = 1j*np.sin(delta/2)*np.exp(-2j*theta) # theta is bounded up to pi
    mat[1,0] = 1j*np.sin(delta/2)*np.exp(2j*theta) # theta is bounded up to pi
    mat[1,1] = np.cos(delta/2)
    
    return mat

# Generates the optic-axis modulation for the waveplate 
# (ang is in degrees)

def rand_optic(n_coeff_x, n_coeff_y, res, maxAng):
    ang = random.uniform(0,maxAng)
    
    x_min = random.uniform(0,res)
    x_max = x_min + random.uniform(0,res)
    y_min = random.uniform(0,res)
    y_max = y_min + random.uniform(0,res)
            
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)

    X, Y = np.meshgrid(x, y)
    
    return np.pi*(generate_random_function(X,Y,0,1,n_coeff_x, n_coeff_y, ang))
    
# Cascade multiple waveplates together, thereby creating our unitary matrix

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

'''
 FUNCTIONS TO RETRIEVE UNITARY PARAMETERS
'''
#  Retrieves En (or big THETA)

def retrieve_En(unitary):
    return np.arccos(0.5*np.trace(unitary))

# Retrieves nx (En must be computed for that unitary) 

def retrieve_nx(unitary, En):
    return (1j/(2*np.sin(En)))*np.trace(unitary*s1)

# Retrieves ny (En must be computed for that unitary) 

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

# Complete function which generates the unitary for one or more waveplates and returns En, theta, phi

def compute_waveplate_cart(num_waveplate, n_coeff, res, maxAng, isCart2):
    
    En = np.zeros([res, res])
    nx = np.zeros([res,res])
    ny = np.zeros([res,res])
    nz = np.zeros([res,res])
    
    theta_list, delta_list = casOptics(num_waveplate, n_coeff, res, maxAng)
    
    for ind1 in range(res):
        for ind2 in range(res):
            # build up the waveplate
            uSyn = np.identity(2, dtype=complex)
            
            for i in range(num_waveplate):
                uSyn = np.dot(uSyn, waveGen(delta_list[i], theta_list[i][ind1][ind2]))
            
            En_pix = retrieve_En(uSyn)
            
            #cartesian
            nx_pix= retrieve_nx(uSyn, En_pix)
            ny_pix =retrieve_ny(uSyn, En_pix)
            nz_pix =retrieve_nz(uSyn, En_pix)
                
            En[ind1,ind2] = En_pix
            nx[ind1,ind2] = nx_pix
            ny[ind1,ind2] = ny_pix
            nz[ind1,ind2] = nz_pix
      
        
    if(isCart2):
          return En, nx, ny
    else: 
          return En, nx, ny, nz
      
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

# These functions are designed to work with the tensorflow architecture during the training process

def UgenTF(En,th,phi): # Compute unitary from parameters - this time we work in tensors
    nx=tf.cast(tf.math.sin(th)*tf.math.cos(phi), dtype=tf.complex64)
    ny=tf.cast(tf.math.sin(th)*tf.math.sin(phi), dtype=tf.complex64)
    nz=tf.cast(tf.math.cos(th), dtype=tf.complex64)
    
    cosEn = tf.cast(tf.math.cos(En), dtype=tf.complex64)
    sinEn = tf.cast(tf.math.sin(En), dtype=tf.complex64)
    
    mat00=cosEn - 1j*sinEn*nz
    mat01=-1j*sinEn*(nx - 1j*ny)
    mat10=-1j*sinEn*(nx + 1j*ny)
    mat11=cosEn + 1j*sinEn*nz
    
    mat = tf.concat([[[mat00,mat01]],[[mat10,mat11]]], 0)
    
    return mat


def compFidTF(mat1, mat2): # Computes the fidelity between two unitaries
    #prod=np.trace(np.dot(np.conjugate(mat1.T),mat2))
    prod=tf.linalg.trace(tf.math.conj(tf.transpose(mat1))@mat2)
    return 0.5*tf.math.abs(prod)
 
def fidReconstructTF(num_pixs,y_true,y_pred,norm=False):
    Fvals = []
    # If normalization is true, modify the datagen so that we spit out modulated values
    if (norm==True):
        normOne = math.pi
        normTwo = 2*math.pi
    else:
        normOne = 1
        normTwo = 1
    # Define function to compute fidelity 
    for i in range(num_pixs):
            for j in range(num_pixs):
                thU = UgenTF(y_true[i,j,0], y_true[i, j,1], y_true[i,j,2])
                expU = UgenTF(y_pred[i,j,0]/normOne, y_pred[i,j,1]/normOne, y_pred[i,j,2]/normTwo)
                Fvals.append(compFidTF(thU,expU))
    Fvals = tf.convert_to_tensor(Fvals)

    return Fvals

def measure(in_pol,out_pol,evo_op,noise): #polarimetric measurement with noise
    
    action=np.dot(evo_op,in_pol)
    
    mel=np.vdot(out_pol,action)
    
    pixNoise = random.gauss(0,noise)
    
    result = 0
    
    if(abs(mel)**2 + pixNoise>1) or (abs(mel)**2 + pixNoise < 0):
        result = abs(mel)**2 - pixNoise 
    else:
        result = abs(mel)**2 + pixNoise 
    
    return result

# State definitions

l=np.array([1,0])
r=np.array([0,1])

h=np.array([1,1])/np.sqrt(2)
v=np.array([1,-1])/np.sqrt(2)

d=np.array([1,1j])/np.sqrt(2)
a=np.array([1,-1j])/np.sqrt(2)

def perturbState(state, randNoise):
    perturbingState = np.array([random.uniform(0, randNoise), random.uniform(0,randNoise)])
    newState = state + perturbingState
    # Enforce normalization 
    normConst = np.sqrt(np.sum(np.abs(newState)**2))
    return newState/normConst

def pixel_measure(En,th,phi,noise, stateNoise):
    
    # Let's introduce slight perturbations onto the states that we choose to measure. 
    # The noise needs to be uncorrelated 
    
    unit_op=Ugen(En,th,phi)
    
    results=np.zeros([5])
      
    results[0]=measure(perturbState(l, stateNoise), perturbState(l, stateNoise), unit_op,noise)
            
    results[1]=measure(perturbState(l, stateNoise), perturbState(h, stateNoise), unit_op, noise)
    
    results[2]=measure(perturbState(l, stateNoise), perturbState(d, stateNoise),unit_op,noise)
            
    results[3]=measure(perturbState(h, stateNoise), perturbState(h, stateNoise),unit_op,noise)
    
    results[4]=measure(perturbState(h, stateNoise), perturbState(d, stateNoise),unit_op,noise)
    
    return results

def pixel_measure_cart(En,nx,ny,nz,noise,stateNoise):
    # Let's introduce slight perturbations onto the states that we choose to measure. 
    # The noise needs to be uncorrelated 
    
    unit_op=Ugen_cart(En,nx,ny,nz)
    results=np.zeros([5])
      
    results[0]=measure(l,l,unit_op,noise)
            
    results[1]=measure(l,h,unit_op,noise)
    
    results[2]=measure(l,d,unit_op,noise)
            
    results[3]=measure(h,h,unit_op,noise)
    
    results[4]=measure(h,d,unit_op,noise)
    
    return results


def full_measure(En,th,phi,res,noise, stateNoise): #measures each pixel and returns flattened res^2 X dim_in vector (dim_in=5)
    
    results=np.zeros([res,res,5])

    for ind1 in range(res):
        for ind2 in range(res):
            results[ind1,ind2,:]=pixel_measure(En[ind1,ind2],th[ind1,ind2],phi[ind1,ind2],noise, stateNoise)
        
    
    return results

def full_measure_cart(En,nx,ny,res,noise,stateNoise, nz=None): # Same function, but now we consider the cartesian parameterization
    if nz is None:
        nz = np.sqrt(np.abs(1-nx**2-ny**2))
    results=np.zeros([res,res,5])
    for ind1 in range(res):
        for ind2 in range(res):
            results[ind1,ind2,:]=pixel_measure_cart(En[ind1,ind2],nx[ind1,ind2], ny[ind1,ind2],nz[ind1,ind2],noise, stateNoise)
            
    return results
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, noise=0.01,stateNoise=0.01, n_coeff=10, res=128, batch_size=100, batches_per_epoch=100, alpha=0.3, n_coeff_low=1, n_coeff_high=15, maxAng=math.radians(10), isWaveplates = True, num_waveplates=10, num_waveplates_min=1, num_waveplates_max=2, unitaryParam = 3, isSingle=False): #default values
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



