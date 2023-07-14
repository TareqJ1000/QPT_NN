# Code that trains different neural network architectures for quantum process tomography

import csv 
import os
import numpy as np
import yaml
from yaml import Loader


import tensorflow as tf
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

# This is code that generates data batch-wise
from dataGenNew import DataGenerator, fidReconstructTF
from UNetArchitecture import uNet
import time

lossObject = BinaryCrossentropy(from_logits=True)

# This is so that we get live feedback on how well our network is learning at each epoch. Credit to this medium article (https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5_)

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    
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
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, 1, figsize=(15,5))
        clear_output(wait=True)
        
        axs.plot(range(1, epoch + 2), 
                    self.metrics['loss'], 
                    label='loss')
        axs.plot(range(1, epoch + 2), 
                    self.metrics['val_loss'], 
                    label='val_loss')
        
        axs.legend()
        axs.grid()
    
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
        
        if (type==2 or type==3 or type==4):
           self.mynn = uNet(size_of_input, type, name)
           #self.mynn.name = name
           
           
                                    
    def forward(self, x): # predicted output of ff_network
        res = self.mynn(x)
        return res
    
# Custom loss function that reconstructs the fidelity from predicted output parameters and 
# computes 1 - average of all fidelities in the batch 

def avg_fidelity_loss(num_pixs):  
    def minMean(y_true, y_pred): 
            
            start = time.time()
            lets_go = tf.map_fn(lambda ind: fidReconstructTF(num_pixs,ind[0],ind[1]), elems=(y_true,y_pred), dtype=(tf.float32, tf.float32), fn_output_signature=tf.float32)
            mean_loss = 1 - tf.reduce_mean((tf.reduce_mean(lets_go)))
            print(mean_loss.dtype)
            return mean_loss
    return minMean 
    

def train_network(config, model, trainGen, validationGen):
    init_lr = eval(config['init_lr']) # staring learn rate
    min_lr = eval(config['min_lr']) # lower bound on learn rate
    epochs_to_update = config['epochs_to_update'] # Number of epochs needed to check condition again 
    lr_factor = config['lr_factor'] # factor by which we modify lr
    num_of_epochs = config['num_of_epochs'] # Number of training epochs
    model_path = config['model_path'] # directory to save the model
    batchSize = config['batchSize'] # batch size PER GRADIENT UPDATE 
    num_pixs = 4
    model_name = config['model_name']
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_path, save_weights_only=False, verbose=1) # callbacks 
    
    adam_optimizer = optimizers.Adam(learning_rate=init_lr)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor = lr_factor, patience = epochs_to_update, min_lr = min_lr, verbose = 1)
    
    model.mynn.compile(loss=avg_fidelity_loss(num_pixs), optimizer=adam_optimizer, metrics = ['accuracy'])
    print("Let us begin with the training!!!")
    history = model.mynn.fit(trainGen,validation_data= validationGen, batch_size=batchSize,  epochs=num_of_epochs, callbacks = [reduce_lr, cp_callback, PlotLearning()])
    # save trained model at the very end
    model_json = model.mynn.to_json()
    with open(model_path + f"/{model_name}.json", 'w') as json_file:
        json_file.write(model_json)
    
    model.mynn.save_weights(model_path+f"/{model_name}.h5")
    print("Saved model to disk")
    
    
    
#####################################################

'''
We can look into this as a secondary loss function if taking the average of fidelities doesn't work out

def custom_loss_entropy(y_true, y_pred):
    return tf.py_function(
        func=crossEntropy, 
        inp = [y_true, y_pred],
        Tout=tf.float64)


def crossEntropy(y_true, y_pred): # Custom loss function that computes the fidelity
    
    num_pixs = len(y_true[0])
    len_batch = len(y_true)
    print(y_true)
    print(y_pred)
    Fvals = np.zeros([len_batch, num_pixs, num_pixs])
    
    # We reconstruct the fidelities pixel by pixel, 
    # the array of fidelity is then compared with 
    # a fidelity of ones. 
    
    for i in range(len_batch):
        for j in range(num_pixs):
            for k in range(num_pixs):
                thU = Ugen(y_true[i,j,k,0], y_true[i, j,k,1], y_true[i,j,k,2])
                expU = Ugen(y_pred[i,j,k,0], y_pred[i,j,k,1], y_pred[i,j,k,2])
                Fvals[i,j,k] = compFid(thU,expU)
                
    # We evaluate the sigmold cross-entropy between the fidelity 
    # and an array of ones
    cust_loss = lossObject(tf.ones_like(Fvals), Fvals)

    return cust_loss
'''


    
    
    

    

