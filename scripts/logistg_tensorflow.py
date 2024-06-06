import decimal
from tabnanny import verbose
import numpy as np
import sklearn
import matplotlib.pyplot  as plt
import tensorflow  as tf
import pandas as pd
import keras
import scipy.sparse as sc
from keras import layers
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import random

from keras.models import model_from_json



from numpy.random import seed


import functions as fc
import re

if __name__ == "__main__":
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    #load data
    #without feature crosses
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_v1-7-GRCh38.npz' 
    trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz' 
    #load loss information
    best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3.npy")
    bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent.npy")


    #full feature model
    #trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/train_v1-7-GRCh38.npz' 
    #trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/test_v1-7-GRCh38.npz' 
    #load loss information
    #best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3_full.npy")
    #bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent_full.npy")


    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False
    #import data (test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    train, Y_train = fc.load_training_data(dense, toDense, trainFile, True)
    train2, Y_train2 = fc.load_training_data(dense, toDense, trainFile2, True)
    
    #Stacking data ontop of each other (splittet in extracting process)
    train = sc.vstack((train,train2))

    Y_train = np.expand_dims(Y_train, axis=1)
    Y_train2 = np.expand_dims(Y_train2, axis=1)
    Y_train =np.vstack((Y_train,Y_train2))

    
    #delet unused data
    del train2
    del Y_train2


    
    # convert scipy sparse matrix to sparse tensor
    X_train = fc.convert_sparse_matrix_to_sparse_tensor(train)

    #get only one autoencoder for training the logistic regression model inside the autoencoder
    best = ["layer_1__latent_807feature_importance_normalize"]

    #early stopping
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)]

    #batch size and number of epochs
    #change epochs to max 50 for model with early stopping
    epo= 13
    batch_size = 256


    for i in best:
       #load autoencoder
        autoencoder = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

        #set autoencoder to trainable flase to only train the weights od the last added perceptron
        autoencoder.trainable =False

        #add the logistic regression to the the model with a batch normalization
        regression = tf.keras.Sequential([
            autoencoder.encoder,
            tf.keras.layers.BatchNormalization(scale=False,center=False),
            tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1))
            #kernel_regularizer regulize weights
            # 1 in perenticies equal to C=1 in sklearn (deafalt and used in CADD)
        ])

        #train the logistic regression model
        #in case of 50 epochs with a erly stopping add the early stopping and change label
        regression.compile(optimizer='adam', loss='binary_crossentropy')
        regression.fit(X_train,Y_train,epochs=epo, batch_size=batch_size, shuffle=True,verbose=2)
        regression.save(filepath='lo_models/tensorflow_'+str(i)+"13_batchnorm")
        

       
            
      
        
        


    
    

    
    
    
    