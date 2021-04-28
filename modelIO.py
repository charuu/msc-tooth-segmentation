
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from niwidgets import NiftiWidget
import tensorflow as tf
from tensorflow.keras.preprocessing.image import * 
from tensorflow import keras
from config import *
from model import *
import math
import random
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from skimage.measure import label
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
from dataAugmentation import *
import cv2
from tensorflow.keras.models import load_model
from skimage.util import random_noise
from skimage import exposure
from tensorflow.keras.losses import Reduction
from tensorflow.keras.losses import Loss
from scipy.ndimage.morphology import distance_transform_edt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer
from matplotlib import pyplot
import tensorflow_addons as tfa


from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
import io
from PIL import Image

def unet_weight_map(y, wc=None, w0 = 1, sigma = 2):
    label_ids = [0,1]
    print(label_ids)
    flat_labels = tf.reshape(y, [-1, 2])

    if len(label_ids) > 1:
        d1 = distance_transform_edt(flat_labels)
        w = -1 * w0 * np.exp(-1/2*((d1) / sigma)**2)
    else:
        w = np.zeros_like(y)
        
        
    if wc:
        class_weights = np.zeros_like(flat_labels)
        for k, v in wc.items():
            class_weights[flat_labels == k] = v
        w = w + class_weights
    return w



class Dice(Loss):
      def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = gen_math_ops.cast(y_true, y_pred.dtype)
        y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
        numerator = (2. * intersection + 1) 
        denominator = (tf.keras.backend.sum(tf.keras.backend.square(y_true),-1) + tf.keras.backend.sum(tf.keras.backend.square(y_pred),-1) + 1)
        return 1 - (numerator/denominator)


class weighted_bce(Loss):
      def call(self, y_true, y_pred, weight1=0.1, weight0=1 ):
       # wmap =unet_weight_map(y, wc=None, w0 = 1, sigma = 2)
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = gen_math_ops.cast(y_true, y_pred.dtype)
        y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        logloss = -(y_true * tf.keras.backend.log(y_pred) * weight0 + (1 - y_true) * tf.keras.backend.log(1 - y_pred) * weight1 )
        return tf.keras.backend.mean( logloss, axis=-1)

class mse(Loss):
      def call(self, y_true, y_pred, weight1=0.1, weight0=1 ):
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = gen_math_ops.cast(y_true, y_pred.dtype)
        y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        return tf.keras.backend.mean(gen_math_ops.square(y_pred - y_true), axis=-1)



def my_loss_fn(y_true, y_pred):
    wc = {
    0: 1, # background
    1: 5  # objects
    }
    
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = gen_math_ops.cast(y_true, y_pred.dtype)
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    w = unet_weight_map(y_true, wc)
    
    bce = y_true * tf.keras.backend.log(y_pred) + (1. - y_true) *tf.keras.backend.log(1 - y_pred)
    wbce=tf.multiply(bce,w)
    return tf.reduce_mean(wbce, axis=-1)  # Note the `axis=-1`

def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


def sliceVolumeImage(vols):
    saveSlicex = []
    saveSlicey = []
    saveSlicez = []
    
    cnt = 0
    for v in vols:
        
        #print(v.shape)
        (dimx, dimy, dimz) = v.shape
        vol = v
        
        if SLICE_X:
            cnt += dimx
            
            for i in range(dimx):
                saveSlicex.append(resize(np.array(vol[i,:,:]),(40,40)))

        if SLICE_Y:
            cnt += dimy
            for i in range(dimy):
                saveSlicey.append(resize(np.array(vol[:,i,:]),(40,40)))

        if SLICE_Z:
            cnt += dimz
            for i in range(dimz):
                saveSlicez.append(resize(np.array(vol[:,:,i]),(40,40)))     
             
        
        slicesVolumeArray = np.rollaxis(np.array([np.concatenate((saveSlicex,saveSlicey,saveSlicez))]),0,4)
      #  print('Number of slices :' + str(cnt))
        
    return slicesVolumeArray



def create_segmentation_generator_train(img, msk, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                         rotation_range=10,
      #                featurewise_center=True
#                      featurewise_std_normalization=True,
#                      rotation_range=30
#                      width_shift_range=0.2,
#                      height_shift_range=0.2,
#                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)
    
    image = sliceVolumeImage(img)
    mask = sliceVolumeImage(msk)
    
    img_generator = datagen.flow(image, batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow(mask, batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img, msk, BATCH_SIZE):
    data_gen_args = dict(rescale=None)
    datagen = ImageDataGenerator(**data_gen_args)
    
    image = sliceVolumeImage(img)
    mask = sliceVolumeImage(msk)
    
    img_generator = datagen.flow(image, batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow(mask, batch_size=BATCH_SIZE, seed=SEED)
    
    return zip(img_generator, msk_generator)



def load(task):
    model= None
    for m in os.listdir(os.path.join('model/UNET3D/',task)):
        with tfmot.sparsity.keras.prune_scope():
            model = load_model(os.path.join('model/UNET3D/',task,m),custom_objects={'Dice': Dice()})
        return model
    return model



def get_model(task):
    if task=='segmentation':
        return unet3d(4)
    if task=='segmentation2d':
        return unet2d(4)
    if task=='classification':
        return classification_net()
    if task=='segmentation_and_classification':
        return segmentation_and_classification(4)
    return None


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    if DATA_AUG==True:
        volume,label = data_aug(volume,label)
    
    volume = tf.expand_dims(volume, axis=3)
    label = tf.expand_dims(tf.math.round(label), axis=3)
    return volume,label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    label = tf.expand_dims(label, axis=3)
    return volume, label


def fixup_shape(images, labels):
    images.set_shape([None,None,None,None])
    labels.set_shape([None,None,None,None]) # I have 19 classes
   
    return images, labels

def show_dataset(data, num,logdir):    
    plt.figure(figsize=(15,15))
 #   file_writer = tf.summary.create_file_writer(logdir)
    for i in range(0,num):  
        images,labels = list(data)[0]
        image = images.numpy()[i]
        label = labels.numpy()[i]
       
    plt.show()
    
def retrain(base_model):
    base_model.trainable = True
    
    x = base_model(base_model.inputs)
    #x = base_model.layers[-3].output
   # x = keras.layers.Dropout(0.2,name="dropafter")(x) 
   # x = keras.layers.Conv3D(1, kernel_size=1, activation='sigmoid', padding='same',name='new_top')(x)
 
    return keras.Model(inputs=[base_model.inputs], outputs=[x], name=f'UNET3D')

def train(task,m,loss,input_arr):
    
    fold_no = 1  
    score=[]
    models =[]
    
   
    name  = '-'.join(['UNET3D-Tooth',task,str(IMAGE_HEIGHT),str(IMAGE_WIDTH),str(IMAGE_DEPTH),datetime.now().strftime("%Y%m%d-%H%M%S"),'.h5'])   
    logdir = LOG_PATH + name
    
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    inputs_train = input_arr[0]
    target_train = input_arr[1]
    inputs_val = input_arr[2]
    target_val = input_arr[3]
    
    print(inputs_train.shape,target_train.shape,inputs_val.shape,target_val.shape)
    
    train_loader = tf.data.Dataset.from_tensor_slices((inputs_train, target_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((inputs_val,target_val))
    
    train_dataset = (
    train_loader
    .shuffle(train_dataset_batch_size)
    .map(train_preprocessing)
    .map(_fixup_shape)
    .batch(train_dataset_batch_size)  
    .prefetch(train_dataset_batch_size)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader
        .map(validation_preprocessing)
        .map(_fixup_shape)
        .batch(test_dataset_batch_size)
        .prefetch(test_dataset_batch_size)
    )
    
    
    if LR == 'CONSTANT':
        lr_schedule = LR_RATE
    else:
        initial_learning_rate = LR_RATE
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100, decay_rate=0.9, staircase=True)
    
    #tf.keras.losses.BinaryCrossentropy()
    if TRAIN==True:
        model= get_model(task)
        loss_fn = Dice()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        model_old = (load(task))
        loss_fn =  Dice()
        
       # model.summary()
   
    #metrics = Metrics()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)
    
    if task =='segmentation2d':
        image = sliceVolumeImage(inputs_train)       
        mask = tf.math.round(sliceVolumeImage(target_train))
        val_image = sliceVolumeImage(inputs_val)       
        val_mask = tf.math.round(sliceVolumeImage(target_val))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss = loss_fn, metrics=['accuracy',tf.keras.metrics.MeanIoU(num_classes=2)])
        model.summary()
        model.fit( image ,mask,validation_data=(val_image,val_mask),
          batch_size=32,
          epochs=NUM_OF_EPOCHS,steps_per_epoch=1,
          callbacks=[earlyStopping,tensorboard_callback])
    else:
        model.summary()
        model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy'],
        )
        
        epochs = NUM_OF_EPOCHS

        model.fit(
            (train_dataset),validation_data=(validation_dataset),
            epochs=epochs,
            shuffle=True,
            callbacks=[earlyStopping,tfmot.sparsity.keras.UpdatePruningStep(),
          tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)],
            )
        
    model.save(PATH +'/'+task+'/' + name)
    
    return model

def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

def predictVolume(model, inImg, toBin=False):
    (xMax, yMax, zMax) = inImg.shape
    
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))
    
    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i,:,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgX[i,:,:] = scaleImg(tmp, yMax, zMax)
    if SLICE_Y:
        cnt += 1.0
        for i in range(yMax):
            img = scaleImg(inImg[:,i,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgY[:,i,:] = scaleImg(tmp, xMax, zMax)
    if SLICE_Z:
        cnt += 1.0
        for i in range(zMax):
            img = scaleImg(inImg[:,:,i], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgZ[:,:,i] = scaleImg(tmp, xMax, yMax)
        
         
    
    outImg = (outImgX + outImgY + outImgZ)/cnt
    #print(outImg.shape)
    if(toBin):
        outImg[outImg>0.5] = 1.0
        outImg[outImg<=0.5] = 0.0
    return outImg





