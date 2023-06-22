# Code designed to train network for QPT

import csv 
import os
import numpy as np
import yaml
from yaml import Loader
import random

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras import optimizers

# This is code that generates data batch-wise
from dataGen import DataGenerator

from neuralnet import ff_network, train_network

stream = open(f"configs/train.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
random.seed(seed)

cnfg['model_name'] +=  f"_{seed}"

# Initializes the model for training 
model = ff_network(6,3,0,cnfg['model_name'])

# Initialize training and validation set 
trainVar = cnfg['params_train']
valVar = cnfg['params_val']


# Initialize training and validation set 
trainGen = DataGenerator(**trainVar)
validationGen = DataGenerator(**valVar)

# With everything in place, let us train the model 
train_network(cnfg, model, trainGen,validationGen)







