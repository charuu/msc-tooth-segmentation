
from skimage.transform import rescale, resize, downscale_local_mean,rotate
from config import * 
from display import *
from skimage import exposure


def resize_data(image,shape=IMG_SIZE):
#rescale(rescale(image, 0.5,anti_aliasing=False), 2,anti_aliasing=False)
    resize_image = resize(image, shape, anti_aliasing=False)
    return (resize_image)



def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
   
    return ((img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE) 


@tf.function
def data_aug(volume,label):

    def rotate(volume,label):
        # define some rotation angles
        angle = ROTATIONANGLE
       # angle = choice(angles)
        volume = rotate(volume, angle)
        label = rotate(label, angle)
      
        return (volume.astype('float64'),label.astype('float64'))
    
    def central_crop(volume,label):
        volume = resize(tf.image.central_crop(volume, 0.75), (IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH), anti_aliasing=False)
        label = resize(tf.image.central_crop(label, 0.75), (IMAGE_HEIGHT, IMAGE_WIDTH,IMAGE_DEPTH), anti_aliasing=False)
      
        return (volume.astype('float64'),label.astype('float64'))
    
    def flip(volume,label):
        if tf.random.uniform(()) > 0.5:
            volume = np.fliplr(volume)
            label =  np.fliplr(label)
        return (volume.astype('float64'),label.astype('float64'))
    
    def flipud(volume,label):
        if tf.random.uniform(()) > 0.5:
            volume = np.flipud(volume)
            label =  np.flipud(label)
        return (volume.astype('float64'),label.astype('float64'))
    
    fns = [flip,flipud,central_crop,rotate]

    from random import choice
    fn= choice(fns)
    #print(fn)
    volume,label = tf.numpy_function( fn, [volume,label], [tf.float64,tf.float64])
    
    return volume,label

def directory(group,folder_type,data_type):
    if group=='train':
        dataDirectory = data_dir_train
        return os.path.join(dataDirectory,folder_type,data_type)
    else:
        dataDirectory = data_dir_test
        return os.path.join(dataDirectory,folder_type,data_type.split('/')[0])
    
    return None

def dataset(data_type,group):
    aug_images = []
    aug_masks= []
    aug_labels=[]
    label = []
    
        
    dir_path_image = directory(group,str('image'),'')

    for niiImage in os.listdir(os.path.join(dir_path_image, data_type)):
        inputs = nib.load( os.path.join(directory(group ,'image',data_type), niiImage)).get_fdata()
        mask = nib.load( os.path.join(directory(group ,'mask',data_type.split('/')[0]), niiImage)).get_fdata() 
        if NORM==True:
            inputs = normalizeImageIntensityRange(inputs) 
        if PREPROCESSING==True:
            inputs = exposure.equalize_hist(inputs)
       # print(data_type)
        if data_type.find('mandible')!=-1 and data_type.find('Molar')!=-1: 
            label=0
        if data_type.find('maxilla')!=-1 and data_type.find('Molar')!=-1: 
            label=1
        if data_type.find('Canine')!=-1 and data_type.find('mandible')!=-1: 
            label=2
        if data_type.find('Incisor')!=-1 and data_type.find('mandible')!=-1:  
            label=3
        if data_type.find('Premolar')!=-1 and data_type.find('mandible')!=-1: 
            label=4
        

        aug_images.append(resize_data(inputs))
        aug_masks.append(np.round(resize_data(mask)))
        aug_labels.append(label)  
        
    return (aug_images,aug_masks,aug_labels)