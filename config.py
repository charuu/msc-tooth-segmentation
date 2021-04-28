import os
import numpy as np
from datetime import *
import math
import random


#config for training network
LR= None
LR_RATE=0.001

TYPE=['mandible-Right-Molar/iteration1','mandible-Right-Molar/iteration4']
iteration ={''}
TRAIN=True
NUM_OF_EPOCHS = 2000
NORM = True
PREPROCESSING= True  
DATA_AUG= True
ROTATIONANGLE = np.random.uniform(-10,10,20) #random.choice([-10, 10])
DROPOUT=0.0
DROPOUT2=0.0

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
IMAGE_DEPTH = 40

IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH)


train_dataset_batch_size = 5
test_dataset_batch_size = 16

SEED = 909

#config for 2d unet segmentation
NUM_TRAIN = 2000
NUM_TEST = 600
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST
#config for 3d unet segmentation

SLICE_X = True
SLICE_Y = True
SLICE_Z = True
HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 4000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

#config path 
TRAINING_MODEL_PATH=''
PATH = 'model/UNET3D'
LOG_PATH= 'logs/UNET3D/' + datetime.now().strftime("%Y%m%d-%H%M%S")
DATA_PATH='/export/skulls/projects/teeth/data/u-net-data'
data_dir = DATA_PATH
data_dir_train = os.path.join(data_dir, 'train/')

data_dir_train_image1 = os.path.join(data_dir_train, 'image1')
data_dir_train_image2 = os.path.join(data_dir_train, 'image2')
data_dir_train_mask1 = os.path.join(data_dir_train, 'mask1')
data_dir_train_mask2 = os.path.join(data_dir_train, 'mask2')

data_dir_test = os.path.join(data_dir, 'test/set2/')
data_dir_test_image = os.path.join(data_dir_test, 'image')
data_dir_test_mask = os.path.join(data_dir_test, 'mask')



