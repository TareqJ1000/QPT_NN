# Code that trains different neural network architectures for quantum process tomography

import os
import numpy as np
import pickle

import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import optimizers

from keras import backend as K

from datagen import DataGenerator, FixedDataGenerator
from UNetArchitecture import uNet
import math 

# This is so that we get live feedback on how well our network is learning at each epoch. Credit to this medium article (https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5_)

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    
    def __init__(self):
        super(PlotLearning,self).__init__()
    
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
            
        model_name = self.model.name
        directory = f'plots/{model_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.tight_layout()
        plt.savefig(directory + f'/epochs_{epoch}.png')
        print('Saved plot of most recent training epoch to disk')
    

# Class which holds the NN model 

class ff_network(tf.Module):
    def __init__ (self, size_of_input, size_of_output, type, name, kernelSize=3, dropRate = 0.1, layers = 1, sixMeasure=False): # This initializes the neural network with a certain chosen architecture
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
           self.mynn = uNet(size_of_input, type, name, kernelSize=kernelSize, dropRate = dropRate, layers=layers, sixMeasure=sixMeasure)
           #self.mynn.name = name
           
    def forward(self, x): # predicted output of ff_network
        res = self.mynn(x)
        return res
    
# Sets of functions which engineers the map fidelity loss function. 

# Returns four N x N x 2 x 2 matrices. Each matrix encodes a different element in a typical 2 x 2 array. 
# batch_size - size of training batch 
# N - image resolution

def get_full_pixelwise(batch_size, N):
    
    part00 = tf.experimental.numpy.full((batch_size,N,N,2,2), [[1,0], [0,0]], dtype=tf.complex64)
    part01 = tf.experimental.numpy.full((batch_size,N,N,2,2), [[0,1], [0,0]], dtype=tf.complex64)
    part10 = tf.experimental.numpy.full((batch_size,N,N,2,2), [[0,0], [1,0]], dtype=tf.complex64)
    part11 = tf.experimental.numpy.full((batch_size,N,N,2,2), [[0,0], [0,1]], dtype=tf.complex64)
    
    return part00, part01, part10, part11

# We compute the matrix elements corresponding to a generic SU(2) operator. Passing the full parameter map directly returns a N x N matrix mapping each parameter map pixel to the corresponding transformation
# En, th, phi -- the spherical unitary parameters

def Ugen_elems(En, th, phi):
    
    # First, convert from spherical to cartesian
    
    nx=tf.cast(tf.math.sin(th)*tf.math.cos(phi), dtype=tf.complex64)
    ny=tf.cast(tf.math.sin(th)*tf.math.sin(phi), dtype=tf.complex64)
    nz=tf.cast(tf.math.cos(th), dtype=tf.complex64)
    
    cosEn = tf.cast(tf.math.cos(En), dtype=tf.complex64)
    sinEn = tf.cast(tf.math.sin(En), dtype=tf.complex64)
    
    # Now, calculate the individual elements 
    
    mat00 = cosEn - 1j*sinEn*nz
    mat01 = -1j*sinEn*(nx-1j*ny)
    mat10 = -1j*sinEn*(nx + 1j*ny)
    mat11 = cosEn + 1j*sinEn*nz
    
    return mat00, mat01, mat10, mat11

# Computes the map fidelity between the true (y_true) and predicted (y_pred) process.
# We optimize, specifically, the map infidelity (1-mapFid)

def mapFidLoss(y_true, y_pred):
    len_batch = tf.shape(y_true)[0]
    res = tf.shape(y_true[0,:,:,0])[0]
    
    # We will modify this to deal with bigger tensorflow size
    
    # For ease of readability, let's formally state what each variable is 
    En_th, theta_th, phi_th = y_true[:,:,:,0], y_true[:,:,:,1], y_true[:,:,:,2]
    En_rec, theta_rec, phi_rec = y_pred[:,:,:,0], y_pred[:,:,:,1], y_pred[:,:,:,2]
    
    # Compute the elements of the theoretical and reconstructed unitary process matrix at each pixel of the image
    mat00_th, mat01_th, mat10_th, mat11_th = Ugen_elems(En_th, theta_th, phi_th)
    mat00_rec, mat01_rec, mat10_rec, mat11_rec = Ugen_elems(En_rec, theta_rec, phi_rec)
    
    # This is towards creating the 2 X 2 matrix on each of the pixels -- we essentially compute |0><0|, |0><1|, |1><0|, and |1><1| for each pixel. 
    Uth_part00, Uth_part01, Uth_part10, Uth_part11 = get_full_pixelwise(len_batch, res)
    Urec_part00, Urec_part01, Urec_part10, Urec_part11 = get_full_pixelwise(len_batch, res)

    # Now, perform elementwise multiplication using these 'part' matrices. This completes the reconstruction of the process matrix at each pixel.  
    Uth_complete = Uth_part00*mat00_th[:,:,:,None,None] + Uth_part01*mat01_th[:, :,:,None,None] + Uth_part10*mat10_th[:, :,:,None,None] +  Uth_part11*mat11_th[:, :,:,None,None]
    Urec_complete = Urec_part00*mat00_rec[:, :,:,None,None] + Urec_part01*mat01_rec[:, :,:,None,None] + Urec_part10*mat10_rec[:, :,:,None,None] +  Urec_part11*mat11_rec[:, :,:,None,None]
    
    # Now compute the pixelwise map fidelity 
    mat1 = tf.math.conj(tf.transpose(Uth_complete, perm = (0,1,2,4,3)))
    mat2 = Urec_complete 
    prod = mat1 @ mat2
    Stot = tf.math.reduce_sum(prod, axis=(1,2))
    tot_trace = tf.linalg.trace(Stot)
    norm_trace = tf.cast(2*res**2, tf.float32)
    
    return 1 - tf.divide(abs(tot_trace), norm_trace)


# (* DEPRECEATED *) Custom metric designed to teach the neural network the cyclicality of our design. Taken from 
# https://stackoverflow.com/questions/37527832/keras-cost-function-for-cyclic-outputs

def mse_cyclic(y_true, y_pred):
    # We assemble the penultimate array elementwise
    part_one = K.mean(K.square(y_pred[:,:,:,0]-y_true[:,:,:,0]))
    part_two = K.mean(K.square(y_pred[:,:,:,1]-y_true[:,:,:,1]))
    part_three = K.mean(
        K.minimum(K.square(y_pred[:,:,:,2]-y_true[:,:,:,2]), 
                  K.minimum(K.square(y_pred[:,:,:,2] - y_true[:,:,:,2] + 2*np.pi), K.square(y_pred[:,:,:,2] - y_true[:,:,:,2] - 2*np.pi))))
    loss_array = tf.stack([part_one, part_two, part_three], axis=0)
    mean_loss = K.mean(loss_array, axis=0)
    
    return mean_loss


# Loads the complete training dataset and splits it up into training and testing according to a train percentange
# filename  -- directory name
# trainPercent -- percentage of data that is assigned for training. 
 
def loadData(filename, trainPercent): 
    
    file = open(filename,'rb')
    X,y = pickle.load(file)
    testPercent = 1 - trainPercent
    
    total = len(X)
    trainLength = int(trainPercent*total)
    testLength =  int(testPercent*total)
    
    # Create shapes for the training/testing dataset
    
    OGShape_X = np.shape(X)
    OGShape_y = np.shape(y)
    
    XShape_train = (trainLength, OGShape_X[1], OGShape_X[2], OGShape_X[3])
    yShape_train =  (trainLength, OGShape_y[1], OGShape_y[2], OGShape_y[3])
    
    XShape_test = (testLength, OGShape_X[1], OGShape_X[2], OGShape_X[3])
    yShape_test = (testLength, OGShape_y[1], OGShape_y[2], OGShape_y[3])

    # Instantiate empty lists for the training/test dataset
    
    X_train = np.empty(XShape_train)
    y_train = np.empty(yShape_train)
    
    X_test = np.empty(XShape_test)
    y_test = np.empty(yShape_test)
    
    # Now, activate the indices
    
    X_train = X[:trainLength]
    y_train = y[:trainLength]
    
    X_test = X[trainLength:]
    y_test = y[trainLength:]

    return X_train, y_train, X_test, y_test


# The main function which prepares the dataset and model for training. 
# config -- configuration file (loaded from the yaml files in the configs folder)
# model - the model to be trained. 

def train_network(config, model):

    # Load (hyper)parameters
    init_lr = eval(config['init_lr']) # staring learn rate
    min_lr = eval(config['min_lr']) # lower bound on learn rate
    epochs_to_update = config['epochs_to_update'] # Number of epochs needed to check condition again 
    lr_factor = config['lr_factor'] # factor by which we modify lr
    num_of_epochs = config['num_of_epochs'] # Number of training epochs
    model_path = config['model_path'] # directory to save the model
    batchSize = config['batchSize'] # batch size PER GRADIENT UPDATE 
    num_pixs = config['num_pixs'] # Resolution of images
    model_name = config['model_name'] # Name of model being trained
    valSplit = config['valSplit'] # Train:Test split (for fixed Dataset)
    enableDataGen = config['enableDataGen'] # Do we use the data generator? 
    
    # Load a pre-trained model (if it exists)
    if (config['load_model']):
        print('loading weights from old data...')
        model.mynn = tf.keras.models.load_model(config['old_model_name'], custom_objects={'math': math, 'mse_cyclic':mse_cyclic, 'mapFidLoss': mapFidLoss}, compile=True)
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # We create a checkpoint of the model at every epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_path, save_weights_only=False, verbose=1) # callbacks 
    
    # Learning rate adapted subject to the validation loss, patience, and factor
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_factor, patience = epochs_to_update, min_lr = min_lr, verbose = 1)
    
    # initialize optimizer and compile model
    adam_optimizer = optimizers.Adam(learning_rate=init_lr)
    model.mynn.compile(loss=mapFidLoss, optimizer=adam_optimizer)
    
    model.mynn.summary()
    print("Let us begin with the training!")
    
    if not enableDataGen: # Generates data in real time
        X_train, y_train, X_test, y_test = loadData(config['datafile'], valSplit)
        shuffle = config['shuffle']
        sigma = config['sigma']
        trainGen = FixedDataGenerator(X_train, y_train, batch_size=batchSize, shuffle=shuffle, sigma=sigma)
        validationGen = FixedDataGenerator(X_test, y_test, batch_size=batchSize, shuffle=shuffle, sigma=sigma)
        history = model.mynn.fit(trainGen, validation_data=validationGen, epochs = num_of_epochs, callbacks = [reduce_lr, cp_callback, PlotLearning])
   
    else: # Loads from a pretrained dataset. 
        trainGen = DataGenerator(**config['train_params'])
        validationGen = DataGenerator(**config['val_params'])
        history = model.mynn.fit(trainGen, validation_data=validationGen,  epochs=num_of_epochs,  callbacks = [reduce_lr, cp_callback, PlotLearning])

    # save trained model at the very end
    model_json = model.mynn.to_json()
    with open(model_path + f"/{model_name}.json", 'w') as json_file:
        json_file.write(model_json)
    
    model.mynn.save_weights(model_path+f"/{model_name}.h5")
    print("Saved model to disk")
    
#####################################################



    
    
    

    

