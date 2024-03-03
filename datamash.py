import numpy as np
import pickle 
import tensorflow as tf

import yaml
from yaml import Loader

import argparse

def loadData(filename): # This supports pickle only for now 
    
    file = open(filename,'rb')
    X,y = pickle.load(file)
    
    return X,y

# parse through slurm array

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

print(shift)

stream = open(f"configs/datamash_simple{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

dataset_1 = cnfg['dataset_1']
dataset_2 = cnfg['dataset_2']
filename = cnfg['fused_dataset_name']

X_1,y_1 = loadData(dataset_1)
X_2,y_2 = loadData(dataset_2)

X_3 = np.concatenate((X_1, X_2)) 
y_3 = np.concatenate((y_1, y_2))

# Dump everything (training examples)

with open(filename + '.pkl', 'wb') as f:
    pickle.dump([X_3, y_3], f)
    
    



 