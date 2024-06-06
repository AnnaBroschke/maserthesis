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
from sklearn import metrics

from keras.models import model_from_json



from numpy.random import seed
import pickle


import functions as fc
import re

if __name__ == "__main__":
    #seeds for all used packages
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)


    #selecting Testsata

    #all variants
    #testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFullReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFullReduced.npz"

    #mAtchet variants without cross features
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleReduced.npz"

    #Downsampled Matched without cross freatures
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleMatchedReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleMatchedReduced.npz"

    #InDel variants
    testFile= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_InFrameInDelsReduced.npz"
    testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_InFrameInDelsReduced.npz"

    #loss information
    #best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3_full.npy")
    #bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent_full.npy")



    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False
    
    #import data (test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    test, y_test = fc.load_training_data(dense, toDense, testFile, True)
    test2, y_test2 = fc.load_training_data(dense, toDense, testFile2, True)


    #Stacking data ontop of each other (splittet in extracting process)
    mat_test = sc.vstack((test,test2))
    y_test = np.expand_dims(y_test, axis=1)
    y_test2 = np.expand_dims(y_test2, axis=1)
    y_test =np.vstack((y_test,y_test2))

    #deleting data not needed anymore
    del test
    del test2
    del y_test2

    #Stacking data ontop of each other (splittet in extracting process)
    b =(y_test==1).flatten()
    c =(y_test==0).flatten()
    test_1 = sc.coo_matrix(mat_test[b])
    test_0 = sc.coo_matrix(mat_test[c])
    
    #deleting not needed data
    del b,c

    
    # convert scipy sparse matrix to sparse tensor
    mat_test = fc.convert_sparse_matrix_to_sparse_tensor(mat_test)
    
    #start figure
    #f, (ax1, ax2) = plt.subplots(2, 1)

    #in case of only two diffrent models othervose use loss information
    labels = ["tensorflow_layer_1__latent_807feature_importance_normalizeearly_stopping8_batchnorm","tensorflow_layer_1__latent_807feature_importance_normalize13_batchnorm"]

    #for loop though all possiple models to print all
    for i in labels:

        #read in autoencoders and logistic regression models
        autoencoder = keras.models.load_model("models_gross/layer_1__latent_807feature_importance_normalize",custom_objects={"sparse_mse" : fc.sparse_mse},)
        model = keras.models.load_model("lo_models/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse, "autoencoder":autoencoder.encoder})

        #predict test data
        y_pred_proba = model.predict(mat_test)

    

        #get metrices on (ROC und AUC)
        fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print(auc)
        #round the AUC to show in plot
        auc1 = np.round(auc, decimals=4)

        
        

        #workaround for plotting diffrent labels
        if i == "tensorflow_layer_1__latent_807feature_importance_normalizeearly_stopping8_batchnorm":


        #Plot results ion plot
            plt.plot(fpr1,tpr1,label="early stopping AUC="+str(auc1))
        else:
            plt.plot(fpr1,tpr1,label="13 epochs AUC="+str(auc1))


    

    #convert Matrixes sparse
    test_1 = fc.convert_sparse_matrix_to_sparse_tensor(test_1)
    test_0 = fc.convert_sparse_matrix_to_sparse_tensor(test_0)
    #load autoencoder
    autoencoder = keras.models.load_model("models_gross/layer_1__latent_807gross",custom_objects={"sparse_mse" : fc.sparse_mse},)

    #transform data into latent space
    latent_test_0 = autoencoder.encoder.predict(test_0, verbose=0)   
    latent_test_1 = autoencoder.encoder.predict(test_1, verbose=0)


    (x1,y1) = latent_test_1.shape
    (x0,y0) = latent_test_0.shape

    #get labes and add both datamatrixes together
    labels_test = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
    latent_space_test = np.vstack((latent_test_1,latent_test_0))
    del latent_test_1,latent_test_0
    del x1,y1,x0,y0

    #scaler
    scaler = pickle.load(open('lo_models/scaler_layer_1__latent_807gross', 'rb'))
    scaler.transform(latent_space_test)


    #load logistic regression model
    modl = pickle.load(open('lo_models/layer_1__latent_807gross', 'rb'))

    #predict test data
    y_pred_proba = modl.predict_proba(latent_space_test)[::,1]

    # get ROC and AUC
    fpr, tpr, _ = metrics.roc_curve(labels_test,  y_pred_proba)
    auc = metrics.roc_auc_score(labels_test, y_pred_proba)
    auc2 = np.round(auc, decimals=4)

    #plot reference curve
    plt.plot(fpr,tpr,label="reference AUC="+str(auc2))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #for diffrent test sets change title
    plt.title('InDel variants')
    plt.tight_layout()
    plt.legend(loc=4)
    #for diffrent test sets change label
    plt.savefig("figures/roc_tensorflow_Indel.svg", bbox_inches='tight')
    

   
    
      
        
        


    
    

    
    
    
    