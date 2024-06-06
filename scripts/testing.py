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

def extract_latent_number(s):
    match = re.search(r'latent_(\d+)', s)
    return int(match.group(1))

if __name__ == "__main__":
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    #Import test data

    #Clinvar without cross features
    #testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFullReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFullReduced.npz"
    #load loss information
    #best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3.npy")
    #bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent.npy")

    #ClinVAR FULL
    testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatFull.npz"
    testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatFull.npz"
    best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3_full.npy")
    bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent_full.npy")

    #ClinVar mit HCdiff
    #testFile = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/XXtest_autoencoderTrainMat.npz"
    #testFile2= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/XXtrain_autoencoderTrainMat.npz"

    #Matched variants full
    #testFile="/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsample.npz" #hat 1213 features
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsample.npz"
    

    #mAtchet variants without cross features
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleReduced.npz"

    #Matched reduced without
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleMatchedReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleMatchedReduced.npz"

    #matched reduces full
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/hallo/train_autoencoderTrainMatDownsampleMatched.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/hallo/test_autoencoderTrainMatDownsampleMatched.npz"

    #neu inDels
    #testFile= "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_InFrameInDelsReduced.npz"
    #testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_InFrameInDelsReduced.npz"

    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False
    
    #import data (test and test 2 are datamatrixes) and (y_train and y_test2) are labels(0=benign,1=patho)
    test, y_test = fc.load_training_data(dense, toDense, testFile, True)
    test2, y_test2 = fc.load_training_data(dense, toDense, testFile2, True)

    print("shape of test")
    print(test.shape)

    print("shape of test2")
    print(test2.shape)


    #Stacking data ontop of each other (splittet in extracting process)
    mat_test = sc.vstack((test,test2))
    y_test = np.expand_dims(y_test, axis=1)
    y_test2 = np.expand_dims(y_test2, axis=1)
    y_test =np.vstack((y_test,y_test2))

    #deleting data not needed anymore
    del test
    del test2
    del y_test2


    #extracting datamatrixes with each label
    b =(y_test==1).flatten()
    c =(y_test==0).flatten()
    test_1 = sc.coo_matrix(mat_test[b])
    test_0 = sc.coo_matrix(mat_test[c])
    

    print("pathogenic anzahl")
    print(sum(b))
    print("benign anzahl")
    print(sum(c))

    del b,c

    #convert Matrixes sparse
    test_1 = fc.convert_sparse_matrix_to_sparse_tensor(test_1)
    test_0 = fc.convert_sparse_matrix_to_sparse_tensor(test_0)
    
    #unify loss information
    best = np.union1d(best3,bestlat)
    best= sorted(best, key=extract_latent_number)

    print("shape of data")
    print(mat_test.shape)
    

    #initialize polot
    f, (ax1, ax2) = plt.subplots(2, 1)

#######################################################################################scaler
    scaler = pickle.load(open('lo_models/scaler_full', 'rb'))
    #scaler = pickle.load(open('lo_models/scaler', 'rb'))
    scaler.transform(mat_test)
    
    #load model without autoencoder
#Change with used data
    #modl2 = pickle.load(open('lo_models/without', 'rb'))
    modl2 = pickle.load(open('lo_models/full_without', 'rb'))

    #predict train data 
    y_pred_proba = modl2.predict_proba(mat_test)[::,1]

    #get metrices on (ROC und AUC)
    fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print(auc)
    auc1 = np.round(auc, decimals=4)

    #delete data which is not used anymore
    del modl2
    del y_test

    #Plot results ion plot
    ax1.plot(fpr1,tpr1,label="without autoencoder AUC="+str(auc1))
    ax1.set_title('best per Latent space')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax1.title.set_text('Best per latent space')
    plt.tight_layout()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

    #plot results in second subplot 
    ax2.plot(fpr1,tpr1,label="without autoencoder AUC="+str(auc1))
    ax2.set_title('top 3')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    ax2.title.set_text('Top 3')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    


    for i in best:
        #get layer and latentspace dimension
        layer = re.search(r'layer_(\d+)', i).group(1)
        lat =int(re.search(r'latent_(\d+)', i).group(1))
      
        #load autoencoder
        autoencoder = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

        #transform data into latent space
        latent_test_0 = autoencoder.encoder.predict(test_0, verbose=0)   
        np.save(file="latent/"+str(i)+"__best_test0",arr=latent_test_0)

        latent_test_1 = autoencoder.encoder.predict(test_1, verbose=0)
        np.save(file="latent/"+str(i)+"__best_test1",arr=latent_test_1)


        (x1,y1) = latent_test_1.shape
        (x0,y0) = latent_test_0.shape

        #get labes and add both datamatrixes together
        labels_test = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
        latent_space_test = np.vstack((latent_test_1,latent_test_0))
        del latent_test_1,latent_test_0
        del x1,y1,x0,y0

############################################################################scaler
        scaler = pickle.load(open('lo_models/scaler_'+str(i), 'rb'))
        scaler.transform(latent_space_test)


        #load logistic regression model
        modl = pickle.load(open('lo_models/'+str(i), 'rb'))

        #predict test data
        y_pred_proba = modl.predict_proba(latent_space_test)[::,1]

        # get ROC and AUC
        fpr, tpr, _ = metrics.roc_curve(labels_test,  y_pred_proba)
        auc = metrics.roc_auc_score(labels_test, y_pred_proba)
        print(i)
        print(auc)
        auc2 = np.round(auc, decimals=4)

        print(i)
        print(autoencoder.summary())
        

        if np.isin(i,best3):
            
            #Plot berst 3 autoencoders in second subplot
            ax2.plot(fpr,tpr,label="latentdim "+str(lat)+" layer "+str(layer)+" AUC="+str(auc2))
            
            ax2.set_ylabel('True Positive Rate')
            ax2.set_xlabel('False Positive Rate')
            ax2.title.set_text('Top 3 autoencoder')
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        #plot ROC in first sublot
        ax1.plot(fpr,tpr,label="latentdim "+str(lat)+" layer "+str(layer)+" AUC="+str(auc2))
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
# change name of saved plot depending on used dastaset and used scaling
        #wihout cross features
        f.savefig("figures/roc_InDels.pdf", bbox_inches='tight')
        #full model
        #f.savefig("figures/roc_best_full_matched.pdf", bbox_inches='tight')
        #matched variants
        #f.savefig("figures/roc_best_matched.pdf", bbox_inches='tight')
       