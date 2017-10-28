import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, Cropping2D, SpatialDropout2D
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "../data/"
LOG_PATH = os.path.join(DATA_PATH, "driving_log.csv")
BATCH_SIZE = 64
EPOCHS = 10

def model_nvidia():

    model = Sequential()
    # Normalize input planes
    model.add(Lambda(lambda x: x/255-0.5, input_shape=(160, 320, 3)))
    # Cropping image for focus only on the road
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # Conv 1 layer
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    # Conv 2 layer
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    # Conv 3 layer
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="relu"))
    model.add(SpatialDropout2D(0.2))
    # Conv 4 layer
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))
    # Conv 5 layer
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    # Fully Connect layer 1
    model.add(Dense(100, activation='relu'))
    # Fully Connect layer 2
    model.add(Dense(50, activation='relu'))
    # Fully Connect layer 3
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(.5))
    # Output
    model.add(Dense(1))
    # lr=0.001
    model.compile(optimizer=Adam(lr=0.0001), loss='mse')
    return model


def training_image_generator(data, batch_size, data_path):
    """
    Training data generator
    """
    while 1:
        batch = get_batch(data, batch_size)
        features = np.empty([batch_size, 160, 320, 3])
        labels = np.empty([batch_size, 1])
        for i, value in enumerate(batch.index.values):
            # Randomly select right, center or left image
            # plt.figure()
            img, steer_ang = get_random_image_and_steering_angle(data, value, data_path)
            img = img.reshape(img.shape[0], img.shape[1], 3)
            # plt.subplot(1,2,1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.imshow(img)
            # plt.title("Original")
            # plt.xlabel("steering: {}".format(steer_ang))
            # Random Translation Jitter
            #img, steer_ang = trans_image(img, steer_ang)
            # Randomly Flip Images
            random = np.random.randint(1)
            if (random == 0):
                img, steer_ang = np.fliplr(img), -steer_ang
            features[i] = img
            labels[i] = steer_ang
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.subplot(1,2,2)
            # plt.imshow(img)
            # plt.title("Flip")
            # plt.xlabel("steering: {}".format(steer_ang))
            # plt.show()
            yield np.array(features), np.array(labels)

def get_random_image_and_steering_angle(data, value, data_path):
    """ 
    Randomly selected right, left or center images and their corrsponding steering angle.
    """ 
    random = np.random.randint(3)
    if (random == 0):
        img_path = data['left'][value].strip()
        shift_ang = .25
    if (random == 1):
        img_path = data['center'][value].strip()
        shift_ang = 0.
    if (random == 2):
        img_path = data['right'][value].strip()
        shift_ang = -.25
    img = get_img_from_path(os.path.join(data_path, img_path))
    # limg = get_img_from_path(os.path.join(data_path, limg_path))
    # limg = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
    # cimg = get_img_from_path(os.path.join(data_path, cimg_path))
    # cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    # rimg = get_img_from_path(os.path.join(data_path, rimg_path))
    # rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(limg)
    # plt.title("Recover from right")
    # plt.xlabel("Steering angle: {}".format(lshift_ang))
    # plt.subplot(1,3,2)
    # plt.imshow(cimg)
    # plt.title("Stay in middle")
    # plt.xlabel("Steering angle: {}".format(cshift_ang))
    # plt.subplot(1,3,3)
    # plt.imshow(rimg)
    # plt.title("Recover from left")
    # plt.xlabel("Steering angle: {}".format(rshift_ang))
    # plt.show()
    #print(img.shape)
    steer_ang = float(data['steering'][value]) + shift_ang
    return img, steer_ang

def get_img_from_path(img_path):
    """
    Get Image from given img path.
    """
    return cv2.imread(img_path)

def get_images(data, data_path):
    """
    Validate data Generator
    """
    while 1:
        for i in range(len(data)):
            img_path = data['center'][i].strip()
            img = get_img_from_path(os.path.join(data_path, img_path))
            img = img.reshape(1, img.shape[0], img.shape[1], 3)
            steer_ang = data['steering'][i]
            steer_ang = np.array([[steer_ang]])
            yield img, steer_ang

def get_batch(data, batch_size):
    """
    Randomly sampled batch_size of data
    """
    return data.sample(n=batch_size)

def trans_image(image, steer):
    """
    Translated image and corrsponding steering angle.
    """
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (320,160))
    return image_tr, steer_ang

## Load driving log csv
csv = pd.read_csv(LOG_PATH, index_col=False)
csv.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

## random all element from csv
total_count = len(csv)
csv.sample(n=total_count)

## Split data 80% for train, 20% for validation 
training_count = int(0.8 * total_count)

training = csv[:training_count].reset_index()
validation = csv[training_count:].reset_index()

## Creating a model
model = model_nvidia()
samples_per_epoch = int(len(training) / BATCH_SIZE) * BATCH_SIZE
nb_val_samples = len(validation)

## Visualize generator data
# featuress = get_images(training, DATA_PATH)
# list(featuress)
# img = featuress.__next__()
# plt.figure()
# print(img[0].shape)
# img = cv2.cvtColor(img[0].squeeze(), cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

## Visualize model architecture
#plot_model(model, to_file='model.png')

values = model.fit_generator(training_image_generator(training, BATCH_SIZE, DATA_PATH), samples_per_epoch=samples_per_epoch, nb_epoch=EPOCHS, validation_data=get_images(validation, DATA_PATH), nb_val_samples=len(validation))

model.save('model.h5')