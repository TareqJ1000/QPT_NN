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

#from tensorflow_addons.layers import Snake
#from typeguard import typechecked
#from tensorflow_addons.activations.snake import snake
#from tensorflow_addons.utils import types

import math

# The encoder and decoder blocks are defined as the HSL - TFP paper

def EncoderBlock(filters, size, layers, middle=False, avgpoolsize=2):
    initializer = tf.random_normal_initializer(0, 0.02)

    conv = Sequential()
    if (middle):
        conv.add(AveragePooling2D(pool_size=(avgpoolsize,avgpoolsize)))
        
    for i in range(layers):
        conv.add(Conv2D(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))
        
    return conv

def DecoderBlock(filters, size, layers, upsamplesize=2):
    
   # initializer = tf.random_normal_initializer(0, 0.02)
    
    conv = Sequential()
    conv.add(UpSampling2D(size=(upsamplesize, upsamplesize)))
    
    for i in range(layers):
        conv.add(Conv2DTranspose(filters, (size,size), padding='same'))
        conv.add(GroupNormalization())
        conv.add(ReLU())
        #conv.add(Snake(name=f'snake_{i}'))

    
    return conv


def uNet(num_pixel, nnType, name, kernelSize=3):
    inputs = Input(shape = [num_pixel, num_pixel, 5])
    isCart = False
    isCart2 = False
    
    if (nnType==2 or nnType==3 or nnType==4): # 16 x 16 
        down_stack = [EncoderBlock(32,kernelSize,1), EncoderBlock(64,kernelSize,1),  EncoderBlock(128,kernelSize,1), EncoderBlock(256,kernelSize,1) ]
        middle = EncoderBlock(512,kernelSize,1,middle=True)
        up_stack = [DecoderBlock(256,kernelSize,1), DecoderBlock(128,kernelSize,1), DecoderBlock(64,kernelSize,1), DecoderBlock(32,kernelSize,1) ]
        
        if(nnType==3):
            isCart = True
        if(nnType==4):
            isCart2 = True
        
        
    if(nnType==5 or nnType==6 or nnType==7): # 32 x 32
        down_stack = [EncoderBlock(32,kernelSize,1), EncoderBlock(64,kernelSize,1),  EncoderBlock(128,kernelSize,1), EncoderBlock(256,kernelSize,1), EncoderBlock(512,kernelSize,1)]
        middle = EncoderBlock(1024,kernelSize,1,middle=True)
        up_stack = [DecoderBlock(512,kernelSize,1), DecoderBlock(256,kernelSize,1), DecoderBlock(128,kernelSize,1), DecoderBlock(64,kernelSize,1), DecoderBlock(32,kernelSize,1)]
        
        if(nnType==6):
            isCart = True
        if(nnType==7):
            isCart2 = True
        
    if(nnType==8 or nnType==9 or nnType==10): # 64 x 64
        down_stack = [EncoderBlock(32,3,1), EncoderBlock(64,3,1),  EncoderBlock(128,3,1), EncoderBlock(256,3,1), EncoderBlock(512,3,1) , EncoderBlock(1024,3,1)  ]
        middle = EncoderBlock(2048,3,1,middle=True)
        up_stack = [DecoderBlock(1024,3,1), DecoderBlock(512,3,1), DecoderBlock(256,3,1), DecoderBlock(128,3,1), DecoderBlock(64,3,1), DecoderBlock(32,3,1) ]
        
        if(nnType==9):
            isCart = True
        if(nnType==10):
            isCart2 = True
    
    if(nnType==11 or nnType==12 or nnType==13): #128 x 128
        down_stack = [EncoderBlock(32,kernelSize,1), EncoderBlock(64,kernelSize,1),  EncoderBlock(128,kernelSize,1), EncoderBlock(256,kernelSize,1),  EncoderBlock(512,kernelSize,1), EncoderBlock(1024,kernelSize,1), EncoderBlock(2048,kernelSize,1)]
        middle = EncoderBlock(4096, kernelSize, 1, middle=True)
        up_stack = [DecoderBlock(2048,kernelSize,1), DecoderBlock(1024,kernelSize,1), DecoderBlock(512,kernelSize,1), DecoderBlock(256,kernelSize,1), DecoderBlock(128,kernelSize,1), DecoderBlock(64,kernelSize,1),  DecoderBlock(32,kernelSize,1)]
        
        if(nnType==12):
            isCart = True
        if(nnType==13):
            isCart2 = True
        
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
    
    if(isCart):
        x = Conv2DTranspose(4, (1,1), padding='same', strides=1, activation='tanh')(x)
        x = tf.keras.layers.Lambda(lambda z: z*[math.pi, 1, 1, 1])(x)
    elif(isCart2):
        x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='tanh')(x)
        x = tf.keras.layers.Lambda(lambda z: z*[math.pi, 1, 1])(x)
    else:
        x = Conv2DTranspose(3, (1,1), padding='same', strides=1, activation='tanh')(x)
        x = tf.keras.layers.Lambda(lambda z: z*[math.pi, math.pi, 2*math.pi])(x)
        
    unet = Model(inputs=inputs, outputs=x, name=name) #  tf.divide(x,normTensors)
    
    return unet
    
# Test out the new function 

#ziggy = uNet(16, 2,'spiffyfaf', kernelSize=3)

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



    

        
            
        
        





