import tensorflow as tf
import numpy as np
import random
import csv
import cv2 
import h5py
from PIL import Image

def get_csv_data(log_file):

    image_names, steering_angles = [], []
    # Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    # Returns ([imagePaths], [measurements])
    correction = 0.275  # steering correction estimate
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+correction, angle-correction])

    return image_names, steering_angles
    

def generator(X_train, y_train, batch_size=64):

    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(batch_size):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            image = cv2.imread(str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[40:-20,:]

            image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)

            image = np.array(image, dtype=np.float32)

            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66,200,3)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model    


if __name__=="__main__":


    X_train, y_train = get_csv_data('UdRec/driving_log.csv')
    X_train, y_train = shuffle(X_train, y_train, random_state=14)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=14)

    # Get model, print summary, and train using a generator
 
    model = get_model()
    model.summary()
    model.fit_generator(generator(X_train, y_train),
                        samples_per_epoch=12234,
                        nb_epoch=5,
                        validation_data=generator(X_validation, y_validation),
                        nb_val_samples=len(X_validation),
                        verbose=1)
 

    print('Saving model weights and configuration file.')
    # Save model weights
    model.save('model.h5')

    # Explicitly end tensorflow session
    from keras import backend as K 

    K.clear_session()
