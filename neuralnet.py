# Code that trains different neural network architectures for quantum process tomography

import os
import numpy as np
import yaml
from yaml import Loader
import pickle
from random import choice

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError

from keras import backend as K

# This is code that generates data batch-wise
from dataGenNew import DataGenerator, fidReconstructTF
from UNetArchitecture import uNet
import time
# This is so that we get live feedback on how well our network is learning at each epoch. Credit to this medium article (https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5_)

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    
    def __init__(self, enableGPU):
        super(PlotLearning,self).__init__()
        self.enableGPU = enableGPU
    
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        print(' Saving current plot of training evolution')
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        # metrics = [x for x in logs if 'val' not in x]
        
        if (self.enableGPU): # we only plot the loss
            f, axs = plt.subplots(1, 1, figsize=(5,5))
            clear_output(wait=True)
            
            axs.plot(range(1, epoch + 2), 
                    self.metrics['loss'], 
                    label='loss')
            axs.plot(range(1, epoch + 2), 
                    self.metrics['val_loss'], 
                    label='val_loss')
    
            axs.legend()
            axs.grid()
            
        else: # Make plots for the whole set of metrics
            f, axs = plt.subplots(1, 3, figsize=(5,5))
            clear_output(wait=True)
        
            axs[0].plot(range(1, epoch + 2), 
                self.metrics['loss'], 
                label='loss')
            axs[0].plot(range(1, epoch + 2), 
                self.metrics['val_loss'], 
                label='val_loss')
            axs[1].plot(range(1,epoch+2), self.metrics['minMean'], label='avg_Fid')
            axs[1].plot(range(1,epoch+2), self.metrics['val_minMean'], label ='avg_fid_val')
            axs[2].plot(range(1,epoch+2), self.metrics['minMeanNormal'], label='avg_Fid_norm')
            axs[0].legend()
            axs[0].grid()
            axs[1].legend()
            axs[1].grid()
            axs[2].legend()
            axs[2].grid()
            
        model_name = self.model.name
        directory = f'plots/{model_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.tight_layout()
        plt.savefig(directory + f'/epochs_{epoch}.png')
        print('Saved plot of most recent training epoch to disk')
    
    

class ff_network(tf.Module):
    def __init__ (self, size_of_input, size_of_output, type, name): # This initializes the neural network with a certain chosen architecture
        super(ff_network, self).__init__()
        
        if (type==0): # This is the original architecture of the network as seen in the paper. 
        
            self.mynn = Sequential([Conv2D(4,(3,3), input_shape=(size_of_input, size_of_input, 5), activation=LeakyReLU(alpha=0.1)),
                                    Conv2D(3,(3,3), activation=LeakyReLU(alpha=0.1)), 
                                    tf.keras.layers.Flatten(), 
                                    Dense(500, activation=LeakyReLU(alpha=0.1)),
                                    Dense(size_of_input*size_of_input*3, activation = 'sigmoid'), Reshape((size_of_input, size_of_input, 3))],
                                   name = name)
            

        if (type==1): # Same as type 0, but we now add max pooling layers in between to help reduce parameter space. 
               
           self.mynn = Sequential([Conv2D(4,(3,3), input_shape=(size_of_input, size_of_input, 5), activation=LeakyReLU(alpha=0.1)),
                                       MaxPooling2D(pool_size=(2,2)),
                                       Conv2D(8,(3,3), activation=LeakyReLU(alpha=0.1)), 
                                       MaxPooling2D(pool_size=(2,2)),
                                       Conv2D(16, (3,3), activation=LeakyReLU(alpha=0.1)), 
                                       MaxPooling2D(pool_size=(2,2)),
                                       tf.keras.layers.Flatten(), 
                                       Dense(500, activation=LeakyReLU(alpha=0.1)),
                                       Dense(size_of_input*size_of_input*3, activation = 'sigmoid'), Reshape((size_of_input, size_of_input, 3))], name=name)
        
        if (type==2 or type==3 or type==4 or type==5 or type==6 or type==7 or type==8 or type==9 or type==10 or type==11 or type==12 or type==13):
           self.mynn = uNet(size_of_input, type, name)
           #self.mynn.name = name
           
           
                                    
    def forward(self, x): # predicted output of ff_network
        res = self.mynn(x)
        return res
    
# Custom loss function that reconstructs the fidelity from predicted output parameters and 
# computes 1 - average of all fidelities in the batch 

def avg_fidelity_loss(num_pixs):  
    def minMean(y_true, y_pred):
            lets_go = tf.map_fn(lambda ind: fidReconstructTF(num_pixs,ind[0],ind[1]), elems=(y_true,y_pred), dtype=(tf.float32, tf.float32), fn_output_signature=tf.float32)
            mean_loss = tf.reduce_mean(lets_go)
            tf.print(mean_loss)
            return mean_loss
    return minMean 

# Custom metric which evaluates the network's performance on an ancillary dataset
# consisting of just normal examples

def avg_norm_loss(num_pixs,X_sup, y_sup, model):  
    def minMeanNormal(y_true, y_pred):
            # Ask the model to make predictions on the supplementary dataset
            y_supPred = model(X_sup)
            # compute and report the mean fidelity reconstruction for this dataset
            lets_go = tf.map_fn(lambda ind: fidReconstructTF(num_pixs,ind[0],ind[1]), elems=(y_sup,y_supPred), dtype=(tf.float32, tf.float32), fn_output_signature=tf.float32)
            mean_loss = tf.reduce_mean(lets_go)
            
            return mean_loss
    return minMeanNormal 

# Custom metric designed to teach the neural network the cyclicality of our design. Taken from 
# https://stackoverflow.com/questions/37527832/keras-cost-function-for-cyclic-outputs


# This applies the cyclic MSE to the last index only (which is meant to be phi). En and theta are non periodic, so they do not get the same treatment
def mse_cyclic(y_true, y_pred): 
    # We assemble the penultimate array elementwise
    part_one = K.mean(K.square(y_pred[0,:,:,0]-y_true[0,:,:,0]))
    part_two = K.mean(K.square(y_pred[0,:,:,1]-y_true[0,:,:,1]))
    part_three = K.mean(
        K.minimum(K.square(y_pred[0,:,:,2]-y_true[0,:,:,2]), 
                  K.minimum(K.square(y_pred[0,:,:,2] - y_true[0,:,:,2] + 1), K.square(y_pred[0,:,:,2] - y_true[0,:,:,2] - 1))))

    return tf.stack([part_one, part_two, part_three], axis=0)


def mse_cyclic_original(y_true, y_pred):
    return K.mean(
        K.minimum(K.square(y_pred-y_true), 
                  K.minimum(K.square(y_pred - y_true + 1), K.square(y_pred - y_true - 1))), axis=-1)
    
def loadData(filename, batch_size, forTrain = True): # This supports pickle only for now 
    
    file = open(filename,'rb')
    X,y = pickle.load(file)
    
    if (forTrain):
    
        bufferSize = len(X)
        
        # Load the dataset from tf 
        
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        dataset = dataset.shuffle(buffer_size=bufferSize) # Apparently, we should set the buffer size to be greater than or equal to the dataset set
        dataset = dataset.batch(batch_size)
        
        return dataset
    
    else:
        return X,y


# Let's create a generator of our data that will load it up batch by batch 

def batch_generator(X, y, batch_size):
    # Get current resolution 
    res = len(X[0])
    # create empty arrays that will hold the batch
    batch_X = np.zeros((batch_size, res, res, 5))
    batch_y = np.zeros((batch_size, res, res, 3))
    
    indices = np.arange(len(X))
    
    while True:
        for i in range(batch_size):
            # choose random index in features
            index=choice(indices)
            batch_X[i] = X[index]
            batch_y[i] = y[index]
            
            yield batch_X, batch_y
    
def train_network(config, model, trainGen, validationGen):
    
    init_lr = eval(config['init_lr']) # staring learn rate
    min_lr = eval(config['min_lr']) # lower bound on learn rate
    epochs_to_update = config['epochs_to_update'] # Number of epochs needed to check condition again 
    lr_factor = config['lr_factor'] # factor by which we modify lr
    num_of_epochs = config['num_of_epochs'] # Number of training epochs
    model_path = config['model_path'] # directory to save the model
    batchSize = config['batchSize'] # batch size PER GRADIENT UPDATE 
    num_pixs = config['num_pixs']
    model_name = config['model_name']
    valSplit = config['valSplit']
    enableGPU = config['enableGPU']
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Load up training dataset
    #train_dataset = loadData(config['datafile_train'], batchSize)
    #X_train, y_train = loadData(config['datafile_train'], batchSize, forTrain=False)

    # Load up validation dataset 
    #test_dataset = loadData(config['datafile_test'], batchSize)
    #X_test, y_test = loadData(config['datafile_test'], batchSize, forTrain=False)
    
    # load up supplementary dataset 
    X_sup, y_sup = loadData(config['datafile_sup'], batchSize, forTrain=False)

    # Create generators for the training and testing dataset 
    #gen_train = batch_generator(X_train, y_train, batchSize)
    #gen_test = batch_generator(X_test, y_test, batchSize)
    
    # Specify the number of steps per epoch
    
    #step_train = int(len(X_train)/batchSize)
    #step_test = int(len(X_test)/batchSize)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_path, save_weights_only=False, verbose=1) # callbacks 
    
    adam_optimizer = optimizers.Adam(learning_rate=init_lr)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_factor, patience = epochs_to_update, min_lr = min_lr, verbose = 1)
    
    # If GPU is enabled, then we only compute the loss of our network
    
    metricList = []
    if (enableGPU==False):
        metricList = [avg_fidelity_loss(num_pixs),avg_norm_loss(num_pixs, X_sup, y_sup, model.mynn)]
    model.mynn.compile(loss=mse_cyclic, optimizer=adam_optimizer, metrics=metricList)
    
    if (config['load_model']):
        print('loading weights from old data...')
        model.mynn.load_weights(config['old_model_name']+'/variables/variables')
    
    model.mynn.summary()
    print("Let us begin with the training!")
    
    # for tensorboard purposes, define log directory and callback
    # log_dir = 'logs'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.mynn.fit(trainGen, validation_data=validationGen,  epochs=num_of_epochs,  callbacks = [reduce_lr, cp_callback, PlotLearning(enableGPU)])
    # save trained model at the very end
    model_json = model.mynn.to_json()
    with open(model_path + f"/{model_name}.json", 'w') as json_file:
        json_file.write(model_json)
    
    model.mynn.save_weights(model_path+f"/{model_name}.h5")
    print("Saved model to disk")
    
#####################################################



    
    
    

    

