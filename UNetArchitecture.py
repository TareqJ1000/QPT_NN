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

#from tensorflow_addons.layers import Snake
#from typeguard import typechecked
#from tensorflow_addons.activations.snake import snake
#from tensorflow_addons.utils import types

import math

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, layers, middle=False, avgpoolsize=2, useDropOut = False, dropRate = 0.1):
    initializer = tf.random_normal_initializer(0, 0.02)

    conv = Sequential()
    if (middle):
        conv.add(AveragePooling2D(pool_size=(avgpoolsize,avgpoolsize)))
        
    for i in range(layers):
        conv.add(Conv2D(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if (useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))
        
    return conv

def DecoderBlock(filters, size, layers, upsamplesize=2, useDropOut=True, dropRate=0.1):
    
   # initializer = tf.random_normal_initializer(0, 0.02)
    
    conv = Sequential()
    conv.add(UpSampling2D(size=(upsamplesize, upsamplesize)))
    
    for i in range(layers):
        conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        if(useDropOut):
            conv.add(Dropout(dropRate))
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))

    return conv


def uNet(num_pixel, nnType, name, kernelSize=3, dropRate = 0.1, layers = 1):
    inputs = Input(shape = [num_pixel, num_pixel, 5])
    isSingle = False
    
    if (nnType==2 or nnType==3): # 16 x 16 
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers) ]
        middle = EncoderBlock(512,kernelSize,layers,middle=True)
        up_stack = [DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(64,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(32,kernelSize,layers) ]
        if(nnType==3):
            isSingle = True
        
    if(nnType==5 or nnType==6): # 32 x 32
        down_stack = [EncoderBlock(32,kernelSize,layers), EncoderBlock(64,kernelSize,layers),  EncoderBlock(128,kernelSize,layers), EncoderBlock(256,kernelSize,layers), EncoderBlock(512,kernelSize,layers)]
        middle = EncoderBlock(1024,kernelSize,layers,middle=True)
        up_stack = [DecoderBlock(512,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,kernelSize,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(64,kernelSize,layers), DecoderBlock(32,kernelSize,layers)]
        if(nnType==6):
            isSingle = True
        
    if(nnType == 7 or nnType==8): # 64 x 64
        down_stack = [EncoderBlock(32,3,layers), EncoderBlock(64,3,layers),  EncoderBlock(128,3,layers), EncoderBlock(256,3,layers), EncoderBlock(512,3,layers) , EncoderBlock(1024,3,layers)  ]
        middle = EncoderBlock(2048,3,layers,middle=True)
        up_stack = [DecoderBlock(1024,3,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(512,3,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(256,3,layers, useDropOut=True, dropRate=dropRate), DecoderBlock(128,3,layers), DecoderBlock(64,3,layers), DecoderBlock(32,3,layers) ]
        if(nnType==8):
            isSingle = True

    
    if(nnType==9 or nnType==10): #128 x 128
        down_stack = [EncoderBlock(32,kernelSize,1), EncoderBlock(64,kernelSize,1),  EncoderBlock(128,kernelSize,1), EncoderBlock(256,kernelSize,1),  EncoderBlock(512,kernelSize,1), EncoderBlock(1024,kernelSize,1), EncoderBlock(2048,kernelSize,1)]
        middle = EncoderBlock(4096, kernelSize, 1, middle=True)
        up_stack = [DecoderBlock(2048,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(1024,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(512,kernelSize,1, useDropOut=True, dropRate=dropRate), DecoderBlock(256,kernelSize,1), DecoderBlock(128,kernelSize,1), DecoderBlock(64,kernelSize,1),  DecoderBlock(32,kernelSize,1)]
        if (nnType==10):
            isSingle = True
        
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
    
    if(isSingle):
        x = Conv2DTranspose(1, (1,1), padding='same', strides=1, activation='sigmoid')(x)
    else:
        x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='sigmoid')(x)
        x = tf.keras.layers.Lambda(lambda z: z*[math.pi, math.pi, 2*math.pi])(x)
        
    unet = Model(inputs=inputs, outputs=x, name=name) #  tf.divide(x,normTensors)
    
    return unet
    
# Test out the new function 

#ziggy = uNet(64, 7,'spiffyfaf', kernelSize=3)

#ziggy.summary()




###### Depreceated Functions ####

# This is a slight modification of the Snake layer implemented in the tensorflow_addons package


'''
class Snake(tf.keras.layers.Layer):
    """Snake layer to learn periodic functions with the trainable `frequency` scalar.

    See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).

    Args:
        frequency_initializer: Initializer for the `frequency` scalar.
    """

    @typechecked
    def __init__(self, frequency_initializer: types.Initializer = "ones", **kwargs):
        super().__init__(**kwargs)
        self.frequency_initializer = tf.keras.initializers.get(frequency_initializer)
        self.frequency = self.add_weight(name=self.name,
            initializer=frequency_initializer, trainable=True
        )

    def call(self, inputs):
        return snake(inputs, self.frequency)

    def get_config(self):
        config = {
            "frequency_initializer": tf.keras.initializers.serialize(
                self.frequency_initializer
            ),
        }
        base_config = super().get_config()
        
        return {**base_config, **config}

'''



    

        
            
        
        





