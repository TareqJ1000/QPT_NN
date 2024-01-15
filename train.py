# Code designed to train network for QPT

import yaml
from yaml import Loader
import random
import math

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib

# This is code that generates data batch-wise
from dataGenNew import DataGenerator

from neuralnet import ff_network, train_network, avg_fidelity_loss

# Load configuration file
stream = open(f"configs/train0.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
random.seed(seed)

cnfg['model_name'] +=  f"_{seed}"
nntype = cnfg['nnType']

kernelSize = cnfg['kernelSize']
dropRate = cnfg['dropRate']
layers = cnfg['layers']
sixMeasure = cnfg['sixMeasure']

freezeLayers = cnfg['freezeLayers']

# Initializes the model for training 
model = ff_network(cnfg['num_pixs'],3,nntype,cnfg['model_name'], kernelSize=kernelSize, dropRate=dropRate, layers=layers, sixMeasure=sixMeasure)

# Let's try performing transfer learning! (If applicable)

if (freezeLayers):
    # Freeze all layers
    for layer in model.mynn.layers:
        layer.trainable = False
    
    freeze_policy_begin = cnfg['freeze_policy_begin']
    freeze_policy_end = cnfg['freeze_policy_end']
    
    # Unfreeze a few ~ approx middle of dataset 
    for ii in range(freeze_policy_begin, freeze_policy_end):
        model.mynn.layers[ii].trainable = True

# With everything in place, let us train the model 
train_network(cnfg, model)