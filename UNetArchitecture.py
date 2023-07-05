# -*- coding: utf-8 -*-
"""
UNet Architecture 

This code encodes a possible U-Net architecture to learn input-output pairs, which is ideal for image-to-image regression
A shameless translation from the U-Net architecture seen in 'Physics-Informaed Convolutional Neural Networks' paper

"""

import tensorflow as tf 

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import GroupNormalization
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, max_pooling=True):
    conv = Sequential()
    conv.add(Conv2D(filters, (size,size), padding='same', strides=2))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    conv.add(Conv2D(filters,(size,size), padding='same', strides=2))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    if (max_pooling):
        conv.add(MaxPooling2D(pool_size=(2,2)))
    return conv

def DecoderBlock(filters, size):
    conv = Sequential()
    conv.add(Conv2DTranspose(filters, (size,size), padding='same', strides=2))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    conv.add(Conv2DTranspose(filters, (size,size), padding='same', strides=2))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    return conv


def uNet(num_pixel, nnType):
    inputs = Input(shape = [num_pixel, num_pixel, 5])
    
    if (nnType==2): # For 128 x 128 data 
    
    # Let's start with 4 blocks?
    
        down_stack = [EncoderBlock(64,3),  EncoderBlock(128,3)] # So each encoder block downsamples the initial image by a factor of 8 
        middle = AveragePooling2D(pool_size=(2,2))
        up_stack = [DecoderBlock(128,3),  DecoderBlock(64,3), DecoderBlock(32, 3)]
    
        # We have one more convolutional layer which shapes the output
        # into the dimensionality we want 
    
        last = Conv2DTranspose(3, (1,1), padding='same', strides=2) # Doubles the dimensionality of the output

        
    if (nnType==3): # 16 x 16 images as proof of principle
         
         down_stack = [EncoderBlock(64,3)]
         middle = AveragePooling2D(pool_size=(2,2))
         up_stack = [DecoderBlock(64,3), DecoderBlock(64,3)]
         last = Conv2DTranspose(3, (1,1), padding='same', strides=1) # Doubles the dimensionality of the output
         
    # We now string together the encoder/decoder blocks 
    
    x=inputs
    
    for down in down_stack:
        x = down(x)
    x = middle(x)
    for up in up_stack: 
        x = up(x)
            
    x = last(x)
    unet = Model(inputs=inputs, outputs=x)
    
    return unet
    
# Test out the new function 
ziggy = uNet(128, 2)
ziggy.summary()




    
    
    
    

        
            
        
        





