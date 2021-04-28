
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from niwidgets import NiftiWidget
import tensorflow as tf
from tensorflow.keras.preprocessing.image import * 
from tensorflow import keras
from config import *
import math
import random
import numpy as np
from skimage.util import montage, crop
from tensorflow.python.client import device_lib
from tensorflow.keras.metrics import MeanIoU,BinaryCrossentropy,Accuracy,Precision,Recall
from skimage.transform import rescale, resize, downscale_local_mean,rotate
from skimage import data, color
from scipy import ndimage
from scipy import misc
from numpy import fliplr
from datetime import *
from sklearn.model_selection import KFold
import tensorflow_model_optimization as tfmot

def unet3d(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
   
    
    inputs =  keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH, in_channels))
    x = inputs
    convpars = dict(kernel_size=kernel_size, padding='same',activation=None)
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x =  keras.layers.Conv3D(initial_features * 2 ** level, **convpars)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.activations.relu( x)
            
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool3D(pooling_size)(x)
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv3DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        
        for _ in range(n_blocks):
            x = keras.layers.Conv3D(initial_features * 2 ** level, **convpars)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.activations.relu( x)
           
            
    x =  keras.layers.Dropout(DROPOUT)(x) 
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv3D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
        
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET3D-L{n_levels}-F{initial_features}')

def unet2d(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):

    
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            x = keras.layers.BatchNormalization()(x)
       
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
 #       x = keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dropout(DROPOUT)(x)         
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

def segmentation_and_classification(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
   
    
    inputs =  keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH, in_channels))
    x = inputs
    convpars = dict(kernel_size=kernel_size, padding='same',activation=None)
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x =  keras.layers.Conv3D(initial_features * 2 ** level, **convpars)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.activations.relu(x)
            
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool3D(pooling_size)(x)
    # upstream
    con=x
    x2=x
    for level in reversed(range(n_levels-1)):
        x2 = keras.layers.Conv3DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x2)
        x2 = keras.layers.Concatenate()([x2, skips[level]])
        
        for _ in range(n_blocks):
            x2 = keras.layers.Conv3D(initial_features * 2 ** level, **convpars)(x2)
            x2 = keras.layers.BatchNormalization()(x2)
            x2 = keras.activations.relu( x2)
           
            
   # x2 =  keras.layers.Dropout(DROPOUT)(x2) 
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    mask = keras.layers.Conv3D(out_channels, kernel_size=1, activation=activation, padding='same',name='sigmoid')(x2)
    
    
    convpars = dict(kernel_size=9, padding='same',activation=None)
   # x = tf.math.multiply(mask , inputs)
    x3 =  keras.layers.Conv3D(16, **convpars)(con)
    x3 =  keras.layers.BatchNormalization()(x3)
    x3 =  keras.activations.relu(x3)
    x3 =  keras.layers.MaxPool3D(pool_size=2)(x3)
    
    x3 =  keras.layers.GlobalAveragePooling3D()(x3)
    x3 =  keras.layers.BatchNormalization()(x3)
    x3 =  keras.layers.Dense(units=16, activation="relu")(x3)
    x3 =  keras.layers.BatchNormalization()(x3)
    x3 =  keras.layers.Dense(units=8, activation="relu")(x3)
    x3 =  keras.layers.BatchNormalization()(x3)
    x3 =  keras.layers.Dense(units=4, activation="relu")(x3)
  #  x3 =  keras.layers.Dropout(0.5)(x3)
    
    outputs =  keras.layers.Dense(units=5, activation="softmax")(x3)

    # Define the model.
    #model = keras.Model(inputs, outputs, name="3dcnn")
    
    return keras.Model(inputs=[inputs], outputs=[mask,outputs], name=f'UNET3D-L{n_levels}-F{initial_features}')


def classification_net():
    """Build a 3D convolutional neural network model."""
    convpars = dict(kernel_size=9, padding='same',activation=None)
    
    inputs = keras.layers.Input(shape=(32, 32,32, 1))
    x =  keras.layers.Conv3D(16, **convpars)(inputs)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.activations.relu(x)
    x =  keras.layers.MaxPool3D(pool_size=2)(x)
    
    x =  keras.layers.GlobalAveragePooling3D()(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dense(units=16, activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dense(units=8, activation="relu")(x)
    x =  keras.layers.BatchNormalization()(x)
    x =  keras.layers.Dense(units=4, activation="relu")(x)
    x =  keras.layers.Dropout(0.5)(x)
    
    outputs =  keras.layers.Dense(units=5, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model