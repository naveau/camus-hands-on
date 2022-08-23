import cv2
from tqdm import tqdm 
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load dataset
def load_ACDC_dataset(images_path, IMG_SIZE):
    
    # Create list of files to load for both images and the corresponding labels
    masks_files  = glob.glob("{}/labels/*.png".format(images_path)) 
    images_files = glob.glob("{}/images/*.png".format(images_path)) 

    # Introduce some randomness in the order the images will be loaded
    images_files.sort()
    masks_files.sort()
    permutation_index = np.random.permutation( len(masks_files))
    images_files_rnd = [images_files[i] for i in permutation_index]
    masks_files_rnd = [masks_files[i] for i in permutation_index]   
    
    # Reading images and corresponding labels
    X = ReadImages(images_files_rnd, size=(IMG_SIZE, IMG_SIZE))
    y = ReadMasks(masks_files_rnd, size=(IMG_SIZE, IMG_SIZE))    
    
    # Return the images (X) and corresponding labels (y) into numpy array 
    return X, y


# Runtime data augmentation
def get_augmented(
    X_train, 
    Y_train, 
    X_val=None,
    Y_val=None,
    batch_size=32, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        #width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=5,
        #zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    )):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=True, seed=seed)
        Y_datagen_val.fit(Y_val, augment=True, seed=seed)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)
        
        return train_generator, val_generator
    else:
        return train_generator

    
# Reading images 
def ReadImages(images_files, size, crop=None):
    X = []
    for index in tqdm(range(len(images_files))):
        image_read = cv2.imread(images_files[index], cv2.IMREAD_GRAYSCALE)
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_LINEAR)
        image_read = image_read / 255.0
        X.append(image_read)
    X = np.asarray(X, dtype=np.float32)
    X = np.expand_dims(X,-1)
    return X

# Reading masks
def ReadMasks(images_files, size, crop=None):
    y = []
    for index in tqdm(range(len(images_files))):
        image_read = cv2.imread(images_files[index], cv2.IMREAD_GRAYSCALE)
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_NEAREST)
        y.append(image_read)
    y = np.asarray(y, dtype=np.int16) 
    y[y==255]=3
    y[y==170]=2
    y[y==85]=1
    y=tf.keras.utils.to_categorical(y)
    return y