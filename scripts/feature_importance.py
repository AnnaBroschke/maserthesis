import decimal
from tabnanny import verbose
import numpy as np
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
from tabulate import tabulate
from numpy.random import seed
import functions as fc
import re
import shap

if __name__ == "__main__":
    # initilize seeds
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

#load data
    #training data for CADD 1.7 without cross features
    trainFile="/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_v1-7-GRCh38.npz"
    trainFile2 ="/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz"

    #test data **+vs.ExAC variants without cross features
    #testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFullReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFullReduced.npz"
    
    #test daset InDel variants without cross features
    testFile= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_InFrameInDelsReduced.npz"
    testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_InFrameInDelsReduced.npz"
    

    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False

    #import data (test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    train, Y_train = fc.load_training_data(dense, toDense, trainFile, True)
    train2, Y_train2 = fc.load_training_data(dense, toDense, trainFile2, True)
    test, y_test = fc.load_training_data(dense, toDense, testFile, True)
    test2, y_test2 = fc.load_training_data(dense, toDense, testFile2, True)
    
    #Stacking training data ontop of each other (splittet in extracting process)
    train = sc.vstack((train,train2))
    Y_train = np.expand_dims(Y_train, axis=1)
    Y_train2 = np.expand_dims(Y_train2, axis=1)
    Y_train =np.vstack((Y_train,Y_train2))
    
    #delete unused data
    del train2
    del Y_train2

    #Stacking test data ontop of each other (splittet in extracting process)
    test = sc.vstack((test,test2))
    y_test = np.expand_dims(y_test, axis=1)
    y_test2 = np.expand_dims(y_test2, axis=1)
    y_test =np.vstack((y_test,y_test2))

    #deleting data not needed anymore
    del test2
    del y_test2

    #get feature names
    names = pd.read_csv("/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/colnames.csv",sep=';', lineterminator='\n')
    names = names.iloc[:461,:1]
    #print(type(names))

    

    #load trained models
    autoencoder = keras.models.load_model("models_gross/layer_1__latent_807feature_importance_normalize",custom_objects={"sparse_mse" : fc.sparse_mse},)
    model = keras.models.load_model("lo_models/tensorflow_layer_1__latent_807feature_importance_normalizeearly_stopping8_batchnorm",custom_objects={"sparse_mse" : fc.sparse_mse, "autoencoder":autoencoder.encoder})

    # get subset of training set for training the shap model
    background = train[np.random.choice(train.shape[0], 1000, replace=False)]
    #  get subset of training set for getting the shap values
    sample = train[np.random.choice(train.shape[0], 5, replace=False)]
    #shap needs dense input 
    sample=sample.todense()
    background = background.todense()
    #print("size of background")
    #print(background.shape)

    #get predictions of the subset
    ped = model.predict(background)

    
    #train shap model
    e = shap.GradientExplainer(model, background)
    #get shap values
    shap_values = e.shap_values(sample)

    #print(shap_values.shape)
    #print(type(shap_values[0]))
    #print(np.shape(shap_values))

    #calculate mean shap value (over 5 samples)

    #initilize means vector
    means = np.ones(461)
    #over all shap values 
    for i in range(461):
        #shap values are saved in a list
        shap_value = shap_values[0]
        
        #calculate means
        means[i] = np.mean(shap_value[:,i])

    # save means in dataframe with names of features
    results = pd.DataFrame({'name':names.iloc[:,0],'shap values':means, 'abs shap values':np.absolute(means)})
    #sort by absolut values of shap values
    results = results.sort_values(by='abs shap values',ascending=False)
    
    #erase absolut values
    results=results.drop(columns=['abs shap values'])
    
    #print out latex table 
    print(tabulate(results, tablefmt="latex_longtable"))
    
    # show histogramm of shap values and save plot
    shap.summary_plot(shap_values, sample, feature_names =names.iloc[:,0].tolist(),max_display=461)
    plt.xlabel('mean SHAP value')
    plt.savefig("figures/feature_Indel_all.pdf", bbox_inches='tight')
   