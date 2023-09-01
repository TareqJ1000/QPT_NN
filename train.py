# Code designed to train network for QPT

import yaml
from yaml import Loader
import random
import math

import tensorflow as tf
import matplotlib.pyplot as plt

# This is code that generates data batch-wise
from dataGenNew import DataGenerator

from neuralnet import ff_network, train_network, avg_fidelity_loss

# Load configuration file
stream = open(f"configs/train2.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

seed = random.randint(1000, 9999)
print(f'seed: {seed}')
random.seed(seed)

cnfg['model_name'] +=  f"_{seed}"
nntype = cnfg['nnType']

# Initializes the model for training 
model = ff_network(cnfg['num_pixs'],3,nntype,cnfg['model_name'])


# Let's try performing transfer learning!!!

# Freeze all layers
for layer in model.mynn.layers:
    layer.trainable = False

freeze_policy = cnfg['freeze_policy']

# Unfreeze a few ~ approx middle of dataset 
for ii in range(freeze_policy, len(model.mynn.layers)):
    model.mynn.layers[ii].trainable = True


# Initialize training and validation set 
trainVar = cnfg['params_train']
valVar = cnfg['params_val']

# Initialize training and validation set 
trainGen = DataGenerator(**trainVar)
validationGen = DataGenerator(**valVar)

# With everything in place, let us train the model 
train_network(cnfg, model)