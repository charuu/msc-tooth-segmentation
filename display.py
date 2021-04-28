import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from niwidgets import NiftiWidget
import tensorflow as tf
from tensorflow.keras.preprocessing.image import * 
from tensorflow import keras
from config import *
from modelIO import *
from model import *
from display import *
from dataAugmentation import *
import math
import random
from skimage.util import montage, crop
from tensorflow.python.client import device_lib
from tensorflow.keras.metrics import MeanIoU,BinaryCrossentropy,Accuracy,Precision,Recall
from skimage.transform import rescale, resize, downscale_local_mean,rotate
from skimage import data, color
from scipy import ndimage
from scipy import misc
from numpy import fliplr
from PIL import Image
from datetime import *
from sklearn.model_selection import KFold

def convert_array_to_image(image_array,filename):
    lower = min(0.0, image_array.min())
    upper = max(1.0, image_array.max())
    montage2 = (image_array - lower) / (upper - lower)
    montage2 = (montage2*255).astype(np.uint8)

    Image.fromarray(montage2).save(filename)
    Image.open(filename)
    
    return Image.open(filename)

def display_monatage(image,mask):
    imag1 = convert_array_to_image(montage(image),'image.png')
    imag2 = convert_array_to_image(montage(mask),'mask.png')
    
    blended = Image.blend(imag2, imag1, alpha=0.50)
   
    fig, ax1 = plt.subplots(1, 1, figsize = (30, 30))
    ax1.imshow(blended, cmap ='gray')
    
    return None
def showDataset(data, num):    
    plt.figure(figsize=(15,15))
    for i in range(0,num):  
        images,labels = list(data)[0]
        image = images.numpy()[i]
        label = labels.numpy()[i]
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(np.squeeze(image[:, :, 23]), cmap="gray")
        ax2.imshow(np.squeeze(label[:, :, 23]), cmap="gray")
        
    plt.show()
    
def display(test_image,test_mask):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(test_image[test_image.shape[0]//2])
    ax1.set_title('Image')
    ax2.imshow(test_mask[test_image.shape[0]//2])
    ax2.set_title('Mask')
       
    display_monatage(test_image,test_mask)
#    display_monatage(np.rollaxis(test_image,0,3),np.rollaxis(test_mask,0,3))
 #   display_monatage(np.rollaxis(test_image,0,2),np.rollaxis(test_mask,0,2))
      
    return None