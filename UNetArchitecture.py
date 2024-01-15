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
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout


# 3D Convolutional layers 
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import UpSampling3D




#from tensorflow_addons.layers import Snake
#from typeguard import typechecked
#from tensorflow_addons.activations.snake import snake
#from tensorflow_addons.utils import types

import math

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, layers, middle=False, avgpoolsize=2, useDropOut = False, dropRate = 0.1, enable3D = False):
    initializer = tf.random_normal_initializer(0, 0.02)

    conv = Sequential()
    if (middle):
        if(enable3D):
            conv.add(AveragePooling3D(pool_size=(avgpoolsize,avgpoolsize, avgpoolsize)))
        else:
            conv.add(AveragePooling2D(pool_size=(avgpoolsize,avgpoolsize)))
            
    for i in range(layers):
        if(enable3D):
            conv.add(Conv3D(filters, (size,size,size), padding='same'))
        else:
            conv.add(Conv2D(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if (useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))
        
    return conv




def DecoderBlock(filters, size, layers, upsamplesize=2, useDropOut=True, dropRate=0.1, enable3D=False):
    
   # initializer = tf.random_normal_initializer(0, 0.02)
    
    conv = Sequential()
    
    if(enable3D):
        conv.add(UpSampling3D(size=(upsamplesize, upsamplesize, upsamplesize)))
    else:
        conv.add(UpSampling2D(size=(upsamplesize, upsamplesize)))
        
    for i in range(layers):
        if(enable3D):
            conv.add(Conv3DTranspose(filters, (size,size,size), padding='same'))
        else:
            conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if(useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))

    return conv

def uNet(num_pixel, nnType, name, kernelSize=3, dropRate = 0.1, layers = 1, sixMeasure = False):
    if (sixMeasure):
        inputs = Input(shape = [num_pixel, num_pixel, 6])
    else:
        inputs = Input(shape = [num_pixel, num_pixel, 5])

    is3D = False
    
    if (nnType==2 or nnType==3): # 16 x 16 
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers) ]
        middle = EncoderBlock(512,kernelSize,layers,middle=True)
        up_stack = [DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(64,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(32,kernelSize,layers) ]
        
    if(nnType==5 or nnType==6): # 32 x 32
        down_stack = [EncoderBlock(32,kernelSize,layers, enable3D=is3D), EncoderBlock(64,kernelSize,layers,  enable3D=is3D),  EncoderBlock(128,kernelSize,layers,  enable3D=is3D), EncoderBlock(256,kernelSize,layers,  enable3D=is3D), EncoderBlock(512,kernelSize,layers,  enable3D=is3D)]
        middle = EncoderBlock(1024,kernelSize,layers,middle=True,  enable3D=is3D)
        up_stack = [DecoderBlock(512,kernelSize,layers, useDropOut=True, dropRate=dropRate,  enable3D=is3D), DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate,  enable3D=is3D), DecoderBlock(128,kernelSize,layers, useDropOut=True, dropRate=dropRate,  enable3D=is3D), DecoderBlock(64,kernelSize,layers,  enable3D=is3D), DecoderBlock(32,kernelSize,layers,  enable3D=is3D)]

        
    if(nnType == 7 or nnType==8): # 64 x 64
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers), EncoderBlock(512,kernelSize,layers) , EncoderBlock(1024,kernelSize,layers)  ]
        middle = EncoderBlock(2048,kernelSize,layers,middle=True)
        up_stack = [DecoderBlock(1024,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(512,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,kernelSize,layers), DecoderBlock(64,kernelSize,layers), DecoderBlock(32,kernelSize,layers) ]

    
    if(nnType==9 or nnType==10): #128 x 128
        down_stack = [EncoderBlock(32,kernelSize,1), EncoderBlock(64,kernelSize,1),  EncoderBlock(128,kernelSize,1), EncoderBlock(256,kernelSize,1),  EncoderBlock(512,kernelSize,1), EncoderBlock(1024,kernelSize,1), EncoderBlock(2048,kernelSize,1)]
        middle = EncoderBlock(4096, kernelSize, 1, middle=True)
        up_stack = [DecoderBlock(2048,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(1024,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(512,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,1), DecoderBlock(128,kernelSize,1), DecoderBlock(64,kernelSize,1),  DecoderBlock(32,kernelSize,1)]

        
    # We now string together the encoder/decoder blocks. This time, we also add skip layers
   
    x=inputs
    
    skips = []
    
    for ii in range(len(down_stack)):
        x = down_stack[ii](x)
        skips.append(x)
        if (ii < len(down_stack)-1):
            if(is3D):
                x = MaxPooling3D(pool_size=(2,2,2))(x)
            else:
                x = MaxPooling2D(pool_size=(2,2))(x)
        
    x = middle(x)
    
    # Reverse skip array 
    skips = reversed(skips)
    
    for up, skip in zip(up_stack,skips): 
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    if (is3D):
        x = Conv3DTranspose(3, (1,1,1), padding='same', strides=1, activation='sigmoid')(x)
    else:
        x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='sigmoid')(x)
        
    if(nnType==5 or nnType==7):
        x = tf.keras.layers.Lambda(lambda z: z*[math.pi, math.pi, 2*math.pi])(x)
        
    unet = Model(inputs=inputs, outputs=x, name=name) #  tf.divide(x,normTensors)
    
    return unet
    
# Test out the new function 

# ziggy = uNet(64, 7, 'spiffyfaf', kernelSize=3, sixMeasure=True)

# ziggy.summary()




        
        





