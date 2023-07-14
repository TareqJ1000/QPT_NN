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
from tensorflow.keras.layers import UpSampling2D

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, max_pooling=True):
    conv = Sequential()
    conv.add(Conv2D(filters, (size,size), padding='same'))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    conv.add(Conv2D(filters,(size,size), padding='same'))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    if (max_pooling):
        conv.add(MaxPooling2D(pool_size=(2,2), strides = 2))
    return conv

def DecoderBlock(filters, size):
    conv = Sequential()
    conv.add(UpSampling2D(size=(2,2)))
    conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
    conv.add(GroupNormalization())
    conv.add(ReLU())
    return conv


def uNet(num_pixel, nnType, name):
    inputs = Input(shape = [num_pixel, num_pixel, 5])
    
    if (nnType==2): # For 128 x 128 data 
    
    # Let's start with 4 blocks?
    
        down_stack = [EncoderBlock(32,3), EncoderBlock(64,3),  EncoderBlock(128,3),  EncoderBlock(256,3), EncoderBlock(512,3), EncoderBlock(512,3)] # So each encoder block downsamples the initial image by a factor of 8 
        middle = AveragePooling2D(pool_size=(2,2))
        up_stack = [DecoderBlock(512,3),  DecoderBlock(512,3), DecoderBlock(256, 3), DecoderBlock(128, 3), DecoderBlock(64, 3), DecoderBlock(32,3) ]

    if (nnType==3): # 16 x 16 images as proof of principle
         
         down_stack = [EncoderBlock(64,3), EncoderBlock(128,3), EncoderBlock(256,3)]
         middle = AveragePooling2D(pool_size=(2,2))
         up_stack = [DecoderBlock(256,3), DecoderBlock(128,3), DecoderBlock(64,3)]
         
    if (nnType==4): # 4x4 images
            down_stack = [EncoderBlock(64,3)]
            middle = AveragePooling2D(pool_size=(2,2))
            up_stack = [DecoderBlock(64,3)]
         

    # We now string together the encoder/decoder blocks. This time, we also add skip layers
    last = Sequential()
    last.add(UpSampling2D(size=(2,2)))
    last.add(Conv2DTranspose(3, (1,1), padding='same', strides=1)) # Doubles the dimensionality of the output
    
    x=inputs
    
    skips = []
    
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    x = middle(x)
    skips.append(x)
    
    # Reverse skip array 
    
    for up, skip in zip(up_stack, skips): 
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
            
    x = last(x)
    unet = Model(inputs=inputs, outputs=x, name=name)
    
    return unet
    
# Test out the new function 
ziggy = uNet(4, 4,'spiffyfaf')
ziggy.summary()


    
    
    

        
            
        
        





