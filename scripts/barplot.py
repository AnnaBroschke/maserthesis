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
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import functions as fc
import re
import pickle

# This skript produces barplots of the AUROC values of each tested model for every tested testset
# in order to produce the plots for the diffrent test sets diffrent lines need to be commented (choosing the wright lines)




#function to get latent space number out of the given name
def extract_latent_number(s):
    match = re.search(r'latent_(\d+)', s)
    return int(match.group(1))

if __name__ == "__main__":
    #initilize seeds
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    #Import test data
    # change diffrent paths to data according on which dataset the test should be

    #Clinvar without cross-features
    #testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFullReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFullReduced.npz"

    #matchet variants without cross-features
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleReduced.npz"

    #downsampled matched variants without cross-features
    testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleMatchedReduced.npz"
    testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleMatchedReduced.npz"

    #InDel variants without cross-features
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_InFrameInDelsReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_InFrameInDelsReduced.npz"



    #ClinVar full features
    #testFilef="/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFull.npz"
    #testFile2f = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFull.npz"

    #Matched variants full features
    #testFilef="/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsample.npz" #hat 1213 features
    #testFile2f = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsample.npz"
    
    #downsampled matched variants with full features
    testFilef= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleMatched.npz"
    testFile2f = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleMatched.npz"

    #InDel variants with full features
    #testFilef= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_InFrameInDelsFull.npz"
    #testFile2f = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_InFrameInDelsFull.npz"

    #load loss info 
    # without cross-features
    best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3.npy")
    bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent.npy")
    # full features
    best3f = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3_full.npy")
    bestlatf = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent_full.npy")


    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False
    
    #import data (test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    test, y_test = fc.load_training_data(dense, toDense, testFile, True)
    test2, y_test2 = fc.load_training_data(dense, toDense, testFile2, True)

    #import data full(test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    testf, y_testf = fc.load_training_data(dense, toDense, testFilef, True)
    test2f, y_test2f = fc.load_training_data(dense, toDense, testFile2f, True)


    #Stacking testdata ontop of each other (splittet in extracting process) without cross-features
    mat_test = sc.vstack((test,test2))
    y_test = np.expand_dims(y_test, axis=1)
    y_test2 = np.expand_dims(y_test2, axis=1)
    y_test =np.vstack((y_test,y_test2))

 
    #Stacking testdata ontop of each other (splittet in extracting process) for full feature space
    mat_testf = sc.vstack((testf,test2f))
    y_testf = np.expand_dims(y_testf, axis=1)
    y_test2f = np.expand_dims(y_test2f, axis=1)
    y_testf =np.vstack((y_testf,y_test2f))

    #deleting data not needed anymore
    del testf
    del test2f
    del y_test2f
    del test
    del test2
    del y_test2


    #extracting datamatrixes with each label to transform them with the autoencoder to get labels into diffrent dimension
    b =(y_test==1).flatten()
    c =(y_test==0).flatten()
    test_1 = sc.coo_matrix(mat_test[b])
    test_0 = sc.coo_matrix(mat_test[c])
    b =(y_testf==1).flatten()
    c =(y_testf==0).flatten()
    test_1f = sc.coo_matrix(mat_testf[b])
    test_0f = sc.coo_matrix(mat_testf[c])
    del b,c

    #convert Matrixes to sparse matrixes 
    test_1 = fc.convert_sparse_matrix_to_sparse_tensor(test_1)
    test_0 = fc.convert_sparse_matrix_to_sparse_tensor(test_0)
    test_1f = fc.convert_sparse_matrix_to_sparse_tensor(test_1f)
    test_0f = fc.convert_sparse_matrix_to_sparse_tensor(test_0f)
    
    #unify loss information
    best = np.union1d(best3,bestlat)
    bestf = np.union1d(best3f,bestlatf)
    

    #load scaler scaler
    scaler = pickle.load(open('lo_models/scaler', 'rb'))
    scaler.transform(mat_test)
    scalerf = pickle.load(open('lo_models/scaler_full', 'rb'))
    scalerf.transform(mat_testf)
    
    #load model without autoencoder
    modl2 = pickle.load(open('lo_models/without', 'rb'))
    modl2f = pickle.load(open('lo_models/full_without', 'rb'))

    #predict train data 
    y_pred_proba = modl2.predict_proba(mat_test)[::,1]
    y_pred_probaf = modl2f.predict_proba(mat_testf)[::,1]

    #get metrices  AUC without autoencoder
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    labels = ["CADD 1.7 reproduced"]
    aucf = metrics.roc_auc_score(y_testf, y_pred_probaf)
    labelsf = ["CADD 1.7 without cross-features"]

    del modl2
    del y_test
    del modl2f
    del y_testf



# going though all autoencoder in the selection to predict the AUROC value for feature space without cross-features

    for i in best:
        #get layer and latentspace dimension from name
        layer = re.search(r'layer_(\d+)', i).group(1)
        lat =int(re.search(r'latent_(\d+)', i).group(1))
      
        #load autoencoder
        autoencoder = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

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

        #load scaler
        scaler = pickle.load(open('lo_models/scaler_'+str(i), 'rb'))
        scaler.transform(latent_space_test)

        #load logistic regression model
        modl = pickle.load(open('lo_models/'+str(i), 'rb'))

        #predict test data
        y_pred_proba = modl.predict_proba(latent_space_test)[::,1]

        # get  AUC
        aucbest = metrics.roc_auc_score(labels_test, y_pred_proba)


        # save name and AUROC value in arrray
        auc =np.append(auc,aucbest)
        labels = np.append(labels,str(layer)+" layer; "+str(lat)+" latentspace dimensions")

# going though all autoencoder in the selection to predict the AUROC value for feature space without cross-features
    for i in bestf:
        
        #load autoencoder
        autoencoderf = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

        #transform data into latent space
        latent_test_0f = autoencoderf.encoder.predict(test_0f, verbose=0)   
        latent_test_1f = autoencoderf.encoder.predict(test_1f, verbose=0)
        
        (x1,y1) = latent_test_1f.shape
        (x0,y0) = latent_test_0f.shape

        #get labes and add both datamatrixes together
        labels_testf = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
        latent_space_testf = np.vstack((latent_test_1f,latent_test_0f))
        del latent_test_1f,latent_test_0f
        del x1,y1,x0,y0

        #scaler
        scaler = pickle.load(open('lo_models/scaler_'+str(i), 'rb'))
        scaler.transform(latent_space_testf)

        #load logistic regression model
        modlf = pickle.load(open('lo_models/'+str(i), 'rb'))

        #predict test data
        y_pred_probaf = modlf.predict_proba(latent_space_testf)[::,1]

        # get  AUC
        aucbestf = metrics.roc_auc_score(labels_testf, y_pred_probaf)

        # get layer and latent space dimension
        layer = re.search(r'layer_(\d+)', i).group(1)
        lat =int(re.search(r'latent_(\d+)', i).group(1))

        #save name and AUROC value in array
        aucf =np.append(aucf,aucbestf)
        labelsf = np.append(labelsf,str(layer)+" layer; "+str(lat)+" latentspace dimension")
        
  
    #sort AUC values
    sorted_indices1 = np.argsort(auc)
    sorted_indices2 = np.argsort(aucf)
    aucf = aucf[sorted_indices2]
    auc = auc[sorted_indices1]
    labelsf = labelsf[sorted_indices2]
    labels = labels[sorted_indices1]

    #get write AUROC value from puplication 
    #select write number for used dataset

    labelpap = ["CADD 1.7"]
    #all variants
    #aucpap = [0.9846]
    #matched variants
    #aucpap = [0.9871093965431392]
    #downsampled matched variants
    aucpap = [0.9877268275536129]
    #InDel variants
    #aucpap = [0.8690912115379215]


    
    

# Plotting


# Plotting first analysis
    plt.barh(labels, auc, color='steelblue', label='without cross features')

# Plotting second analysis
    plt.barh(labelsf, aucf, color='yellowgreen', label='full feature space')

# Adding dashed line to separate the two analyses
    plt.axhline(y=len(labels)-0.5, color='black', linestyle='--')

    plt.barh(labelpap, aucpap, color='firebrick', label='published')

    plt.axhline(y=len(labels)+len(labelsf)-0.5, color='black', linestyle='--')

# Adding labels and title
    plt.xlabel('AUC Values')
    #change title according to used variants
    plt.title('downsampled matched variants')
    plt.legend(loc=3)
    plt.tight_layout()
    #change name of the saving image according to used variants
    plt.savefig("figures/auc_bar_matched_down.svg", bbox_inches='tight')

    