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
from tensorflow.math import sigmoid
import math

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, layers, middle=False):
    initializer = tf.random_normal_initializer(0, 0.02)

    conv = Sequential()
    if (middle):
        conv.add(AveragePooling2D(pool_size=(2,2)))
    for i in range(layers):
        conv.add(Conv2D(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        conv.add(ReLU())
    return conv

def DecoderBlock(filters, size, layers):
    
    initializer = tf.random_normal_initializer(0, 0.02)
    
    conv = Sequential()
    conv.add(UpSampling2D(size=(2,2)))
    
    for i in range(layers):
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
            down_stack = [EncoderBlock(64,3,1), EncoderBlock(128,3,1)]
            middle = EncoderBlock(128,3,1,middle=True)
            up_stack = [DecoderBlock(128,3,1), DecoderBlock(64,3,1) ]

    if (nnType==5): # 16 x 16 images -- higher resolution might mean more information 
               down_stack = [EncoderBlock(32,3,1), EncoderBlock(64,3,1),  EncoderBlock(128,3,1), EncoderBlock(256,3,1) ]
               middle = EncoderBlock(512,3,1,middle=True)
               up_stack = [DecoderBlock(256,3,1), DecoderBlock(128,3,1), DecoderBlock(64,3,1), DecoderBlock(32,3,1) ]

               
    # We now string together the encoder/decoder blocks. This time, we also add skip layers
   
    x=inputs
    
    skips = []
    
    for ii in range(len(down_stack)):
        x = down_stack[ii](x)
        skips.append(x)
        if (ii < len(down_stack)-1):
            x = MaxPooling2D(pool_size=(2,2))(x)
        
    x = middle(x)
    
    # Reverse skip array 
    skips = reversed(skips)
    
    for up, skip in zip(up_stack,skips): 
        print(skip)
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    #x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='sigmoid')(x)

    #x = sigmoid(x)
    x = tf.keras.layers.Lambda(lambda z: z*[math.pi, math.pi, 2*math.pi])(x)
    
    unet = Model(inputs=inputs, outputs=x, name=name) #  tf.divide(x,normTensors)
    
    return unet
    
# Test out the new function 
#ziggy = uNet(16, 5,'spiffyfaf')
#ziggy.summary()




    
    
    

        
            
        
        





