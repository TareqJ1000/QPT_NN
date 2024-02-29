# Code designed to train network for QPT

import yaml
from yaml import Loader
import random
import math

import tensorflow as tf
import matplotlib.pyplot as plt

# This is code that generates data batch-wise
from dataGenNew import DataGenerator

from neuralnet import ff_network, train_network

# Load configuration file
stream = open(f"configs/train.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Initialize starting seed. 
seed = random.randint(1000, 9999)
print(f'seed: {seed}')
random.seed(seed)

# Load model parameters
cnfg['model_name'] +=  f"_{seed}" 
nntype = cnfg['nnType'] # select type of network architecture
kernelSize = cnfg['kernelSize'] # size of convolutional kernels
dropRate = cnfg['dropRate'] # Dropout regularization alpha
layers = cnfg['layers'] # Number of convolutional layers per level
sixMeasure = cnfg['sixMeasure'] # Enable six measurements as input? 
freezeLayers = cnfg['freezeLayers'] # Do we (partially) freeze layers of the network during training? 

# Initializes the model for training 
model = ff_network(cnfg['num_pixs'],3,nntype,cnfg['model_name'], kernelSize=kernelSize, dropRate=dropRate, layers=layers, sixMeasure=sixMeasure)

# Let's try performing transfer learning (If applicable)
if (freezeLayers):
    # Freeze all layers
    for layer in model.mynn.layers:
        layer.trainable = False
    
    freeze_policy_begin = cnfg['freeze_policy_begin'] # Starting layer where we do NOT apply freezing
    freeze_policy_end = cnfg['freeze_policy_end'] # Final layer where we do NOT apply freezing
    
    # All layers within the 'freezing policy' are put as unfrozen. 
    for ii in range(freeze_policy_begin, freeze_policy_end):
        model.mynn.layers[ii].trainable = True

# With everything in place, let us train the model 
train_network(cnfg, model)