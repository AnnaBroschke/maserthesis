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
import pickle


from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import functions as fc
import re

if __name__ == "__main__":
    seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    #load data
    #without cross features
    #trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_v1-7-GRCh38.npz' 
    #trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz' 
    #load loss information
    #best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3.npy")
    #bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent.npy")


    #with full feature space
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/train_v1-7-GRCh38.npz' 
    trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/test_v1-7-GRCh38.npz' 
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

    # split up data depending on the label to transform label into latent space
    a =(Y_train==1).flatten()
    b =(Y_train==0).flatten()
    train_1 = sc.coo_matrix(train[a])
    train_0 = sc.coo_matrix(train[b])
    #delet unused data
    del a,b


    
    # convert scipy sparse matrix to sparse tensor
    X_train = fc.convert_sparse_matrix_to_sparse_tensor(train)
    X_train_1 = fc.convert_sparse_matrix_to_sparse_tensor(train_1)
    X_train_0 = fc.convert_sparse_matrix_to_sparse_tensor(train_0)
    
    #delet unused data
    del train_1
    del train_0

    #unify loss information
    #best = ["layer_2__latent_1820gross_full"]
    best = np.union1d(best3,bestlat)


  

    #scaler
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False, copy=False)
    scaler.fit_transform(train)

    #train model without autoencoder
    modl2 = LogisticRegression(penalty='l2', C=1, max_iter=13, solver='lbfgs', warm_start=True, verbose=0)
    modl2.fit(train,Y_train)


    #save logistic regression model, and scaler
    # change name dependnding on used data
    pickle.dump(modl2, open('lo_models/without_full', 'wb'))
    pickle.dump(scaler,open('lo_models/scaler_full', 'wb'))
    
    #delete unused data
    #del modl2
    del Y_train
    del train
  

    #going through all latent space dimension with best autoencoder
    for i in best:
       #load autoencoder
        autoencoder = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

        #extracting the latent space
        latent_label_1 = autoencoder.encoder.predict(X_train_1, verbose=0)
        #np.save(file="latent/"+str(i)+"__best1",arr=latent_label_1)
        latent_label_0 = autoencoder.encoder.predict(X_train_0, verbose=0)
        #np.save(file="latent/"+str(i)+"__best0",arr=latent_label_0)

        # #get labes and add both datamatrixes together
        (x1,y1) = latent_label_1.shape
        (x0,y0) =latent_label_0.shape

        labels_train = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
        latent_space_train = np.vstack((latent_label_1,latent_label_0))
        del latent_label_1,latent_label_0

        #scaler
        scaler.fit_transform(latent_space_train)
        pickle.dump(scaler,open('lo_models/scaler_'+str(i), 'wb'))

        #train logistic regression model
        modl = LogisticRegression(penalty='l2', C=1, max_iter=13, solver='lbfgs', warm_start=True, verbose=0).fit(latent_space_train,labels_train)

       #save logistic regression model
        pickle.dump(modl, open('lo_models/'+str(i), 'wb'))
        


        
            
      
        
        


    
    

    
    
    
    