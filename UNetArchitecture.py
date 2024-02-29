"""

UNet Architecture 
This code encodes a possible U-Net architecture to learn input-output pairs, which is ideal for image-to-image regression

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

import math

# Define encoding/decoding layers/blocks as described in the paper
# (For both functions)
# filters - # of filters/kernels/features
# size - kernel/filter/feature size
# layers - number of convolutional layers in encoding layer
# middle - are we at the bottomost layer of the u-net?
# avgpoolsize - kernel size of the average pooling layer
# useDropOut - do we enable Dropout regularization? 
# dropRate - rate at which Dropout is applied

def EncoderBlock(filters, size, layers, middle=False, avgpoolsize=2, useDropOut = False, dropRate = 0.1):
    # initializer = tf.random_normal_initializer(0, 0.02) (this was in our tests ... check w/o this)
    
    conv = Sequential()
    if (middle):
        conv.add(AveragePooling2D(pool_size=(avgpoolsize,avgpoolsize)))
            
    for i in range(layers):
        conv.add(Conv2D(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if (useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())
        
    return conv

def DecoderBlock(filters, size, layers, upsamplesize=2, useDropOut=True, dropRate=0.1, enable3D=False):

    conv = Sequential()
    conv.add(UpSampling2D(size=(upsamplesize, upsamplesize)))
        
    for i in range(layers):
        conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if(useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())

    return conv

# Creates the U-Net architecture 
# num_pixel - image resolution 
# nnType - type of architecture to be constructed
# name - model name
# kernelSize - size of convolutional kernel 
# dropRate - value for Dropout rate
# layers - number of convolutional layers per encoding/decoding 'block' 
# sixMeasure - do we enable six measurements as input? 

def uNet(num_pixel, nnType, name, kernelSize=3, dropRate = 0.1, layers = 1, sixMeasure = False):
    if (sixMeasure):
        inputs = Input(shape = [num_pixel, num_pixel, 6])
    else:
        inputs = Input(shape = [num_pixel, num_pixel, 5])


    if(nnType==0): # 64 x 64
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers), EncoderBlock(512,kernelSize,layers) , EncoderBlock(1024,kernelSize,layers)  ]
        middle = EncoderBlock(2048,kernelSize,layers,middle=True)
        up_stack = [DecoderBlock(1024,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(512,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,kernelSize,layers), DecoderBlock(64,kernelSize,layers), DecoderBlock(32,kernelSize,layers) ]

    
    if(nnType==1): # 128 by 128. We keep this as an example on how our architecture can be scaled 
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers),  EncoderBlock(512,kernelSize,layers), EncoderBlock(1024,kernelSize,layers), EncoderBlock(2048,kernelSize,layers)]
        middle = EncoderBlock(4096, kernelSize, layers, middle=True)
        up_stack = [DecoderBlock(2048,kernelSize, layers, useDropOut=True, dropRate=dropRate), DecoderBlock(1024,kernelSize, layers, useDropOut=True, dropRate=dropRate), DecoderBlock(512,kernelSize, layers, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,layers), DecoderBlock(128,kernelSize,layers), DecoderBlock(64,kernelSize,layers),  DecoderBlock(32,kernelSize,layers)]

        
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
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    
    # Convert and scale to proper dimensions/range
    
    x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='sigmoid')(x)
    x = tf.keras.layers.Lambda(lambda z: z*[math.pi, math.pi, 2*math.pi])(x)
        
    unet = Model(inputs=inputs, outputs=x, name=name) 
    
    return unet
    



        
        





