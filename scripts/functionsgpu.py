import numpy as np
#import sklearn
#import matplotlib.pyplot  as plt
import tensorflow  as tf
#import pandas as pd
import tensorflow.keras
from scipy.sparse import load_npz
from tensorflow.keras import layers
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



#methods taken from https://github.com/kircherlab/CADD/blob/master/scripts/sklearnTools/handling.py
#to load in training and test files
def _load_file(dense, filename):

    if dense:
        mat = np.load(filename)
        y = mat[:,0].reshape((mat.shape[0],))
    else:
        mat = load_npz(filename)
        y = mat[:,0].toarray().reshape((mat.shape[0],))

    return (mat[:,1:], y)

def load_training_data(dense, toDense, trainFile, verbose=True):
    '''
    load the trainin data set and separate label and data
    '''
    

    mat_train, y_train = _load_file(dense, trainFile)

    if verbose:
        print('Training data loaded')

    if not dense and toDense:
        
        # default .todense does not work
        mat_train_dense = np.zeros(mat_train.shape, dtype=mat_train.dtype)

        for i in range(mat_train.shape[0]):
            start = mat_train.indptr[i]
            end = mat_train.indptr[i+1]
            mat_train_dense[i,mat_train.indices[start:end]] = mat_train.data[start:end]

        mat_train = mat_train_dense
        if verbose:
            print('Converted input data to dense format')

    if verbose:
        print('# Training-Samples: %i' % mat_train.shape[0])
        print('# Features: %i\n' % mat_train.shape[1])

    return (mat_train, y_train)

#convert sparse scipy matrix to sparce tensor
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

#define autoencoder with two dense layers
class Autoencoder2(Model):
        def __init__(self, latent_dim,layer2_dim,activation_en,activation_de, shape,droprate):
            super(Autoencoder2, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=shape,sparse=True),
                layers.Lambda(tf.sparse.to_dense),
                layers.Dense(layer2_dim, activation=activation_en),
                layers.Dropout(rate=droprate, seed=2020),
                layers.Dense(latent_dim, activation=activation_en),
            ])
            self.decoder = tf.keras.Sequential([
                #layers.Dense(tf.math.reduce_prod(shape), activation=activation_de,activity_regularizer=regularizers.l1(l1=regu))
                layers.Dense(layer2_dim, activation=activation_de),
                layers.Dropout(rate=droprate, seed=2020),
                layers.Dense(tf.math.reduce_prod(shape), activation_de)
                #layers.Reshape(shape)
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

#define autoencoder with one dense layer
class Autoencoder1(Model):
        def __init__(self, latent_dim,layer2_dim,activation_en,activation_de, shape,droprate):
            super(Autoencoder1, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=shape,sparse=True),
                layers.Lambda(tf.sparse.to_dense),
                #layers.Dropout(rate=droprate, seed=2020),
                layers.Dense(latent_dim, activation=activation_en)
                
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dropout(rate=droprate, seed=2020),
                layers.Dense(tf.math.reduce_prod(shape), activation=activation_de)
                
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

# Custom loss function for handling sparse true values
def sparse_mse(y_true, y_pred):
    # Check if y_true is a sparse tensor and convert it to dense
    if isinstance(y_true, tf.SparseTensor):
        y_true_dense = tf.sparse.to_dense(y_true)
    else:
        y_true_dense = y_true

    # Compute the mean squared error between the true and predicted values
    return tf.reduce_mean(tf.square(y_true_dense - y_pred))
