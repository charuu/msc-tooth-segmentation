
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
#print(device_lib.list_local_devices())
#tf.config.experimental.list_physical_devices('GPU')
import time 
import numpy as np
from skimage.io import imshow
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
from config import *
from sklearn.model_selection import *

#Keras

def load_data(task, model,TYPE):
    input_arr1= []
    input_arr2= []
    input_arr3= []
    input_arr4= []
    
    for y in TYPE:
        for t in os.listdir(os.path.join(data_dir_train+str('image'))):
            if y.find(t) != -1:
                d = dataset(y,'train')
                x_train=d[0]
                y_train=d[1]
                input_arr1.append(x_train)
                input_arr2.append(y_train)
                    
   # for y in TYPE:    
    for t in os.listdir(os.path.join(data_dir_test+str('image'))): 
        #if y.find(t) != -1:
           # print(t)
        d1 = dataset(t,'test')
        x_val=d1[0]
        y_val=d1[1]
        input_arr3.append(x_val)
        input_arr4.append(y_val)
        
    a=  np.array(np.concatenate(input_arr1,axis=0))
    b=  np.array(np.concatenate(input_arr2,axis=0))
    c=  np.array(np.concatenate(input_arr3,axis=0))
    d=  np.array(np.concatenate(input_arr4,axis=0))
    
    return (a,b,c,d)

def train_segmentation(task, model,TYPE):
    
    (a,b,c,d) = load_data(task, model)
    
    best_model = train(task, model,'None', (a,b,c,d))

    return best_model



