# Packages
import numpy as np
from scipy.misc import imresize
import keras.backend as K
from keras import losses,metrics
from keras.models import load_model

def rmse(y_true,y_pred):
    '''Accepts true labels and predictions. Returns Root mean squared error'''
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

def load_galaxy_model():
    '''Loads stored keras model with rmse loss and metric'''
    #Before prediction
    K.clear_session()

    # Uses custom loss/metrics
    losses.rmse = rmse
    metrics.rmse = rmse

    return load_model(f'./app/galaxy_morphology_predictor.h5')

def process_image(image):
    '''Accepts an image of any shape and reshapes for use in model predictions'''
    
    # Resize and Reshape Image
    image = imresize(image,(169,169))
    image = image.reshape((1,)+image.shape)
    
    # Scale Image
    means = np.load(f'./app/means.npy')
    stds = np.load(f'./app/stds.npy')

    return (image - means)/stds