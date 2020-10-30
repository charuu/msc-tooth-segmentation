import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

TRAINING_MODEL_PATH=''
DATA_PATH='/export/skulls/projects/teeth/data/u-net-data'
SEED = 909
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
IMAGE_DEPTH = 80
NUM_TRAIN = 400
BATCH_SIZE_TEST =8
ROTATIONANGLE=8
NUM_TEST = 200
EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST
    
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH)
SLICE_X = True
SLICE_Y = True
SLICE_Z = True

data_dir = DATA_PATH
data_dir_train = os.path.join(data_dir, 'train/')
data_dir_train_image = os.path.join(data_dir_train, 'image/image')
data_dir_train_mask = os.path.join(data_dir_train, 'mask/mask')

data_dir_test = os.path.join(data_dir, 'test/')
data_dir_test_image = os.path.join(data_dir_test, 'image/image')
data_dir_test_mask = os.path.join(data_dir_test, 'mask/mask')

NUM_TRAIN = 360
NUM_TEST = 100

NUM_OF_EPOCHS = 100
SCALE = {None,0.5,0.25,2.0}
ROTATIONANGLE = {None, 10}