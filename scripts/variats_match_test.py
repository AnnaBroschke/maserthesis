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

if __name__ == "__main__":
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
# path to data 
    #first to test all pipelines second is the real deal
    # path to data 
    #first to test all pipelines second is the real deal
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_v1-7-GRCh38.npz' 
    trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz' 
    testFile="/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_ClinVarExacSparseMatrixGen.npz"
    testFile2 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_ClinVarExacSparseMatrixGen.npz"


    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False
    #import data mat_train is data and y_train are labels(0=,1=)
    train, Y_train = fc.load_training_data(dense, toDense, trainFile, True)
    test, Y_train2 = fc.load_training_data(dense, toDense, trainFile2, True)
    #import data mat_train is data and y_train are labels(0=,1=)
    mat_test, y_test = fc.load_training_data(dense, toDense, testFile, True)
    mat_test2, y_test2 = fc.load_training_data(dense, toDense, testFile2, True)
    
    mat_test = sc.vstack((mat_test,mat_test2))
    y_test = np.expand_dims(y_test, axis=1)
    y_test2 = np.expand_dims(y_test2, axis=1)
    y_test =np.vstack((y_test,y_test2))

    train = np.vstack(train,test)
    Y_train = np.expand_dims(Y_train, axis=1)
    Y_train2 = np.expand_dims(Y_train2, axis=1)
    Y_train =np.vstack((Y_train,Y_train2))
    #print(np.all(bes[1:10000]==y_test[1:10000]))
    

    
    del mat_test2
    del y_test2
    del test
    del y_test2

    print("size training")
    print(train.shape)
    print("size of testing")
    print(mat_test.shape)
    #splitting test and traindata for validation of the autoencoder
    #train, test, Y_train, Y_tests = train_test_split(mat_train, y_train, train_size = 0.8, random_state = seed(2020))


    train_1 = train[Y_train==1]
    train_0 = train[Y_train==0]
    b =(y_test==1).flatten()
    c =(y_test==0).flatten()
    test_1 = sc.coo_matrix(mat_test[b])
    test_0 = sc.coo_matrix(mat_test[c])

    
    # convert scipy sparse matrix to sparse tensor
    X_train = fc.convert_sparse_matrix_to_sparse_tensor(train)
    X_train_1 = fc.convert_sparse_matrix_to_sparse_tensor(train_1)
    X_train_0 = fc.convert_sparse_matrix_to_sparse_tensor(train_0)
    test_1 = fc.convert_sparse_matrix_to_sparse_tensor(test_1)
    test_0 = fc.convert_sparse_matrix_to_sparse_tensor(test_0)
    #Y_test =convert_sparse_matrix_to_sparse_tensor(test_label[:1000]) 
    #später import datei mit labels



    #input of autoencoder 
    shap = X_train.shape[1:]
    ac_en='leaky_relu'
    ac_de='leaky_relu'
    droprate=0.1
    
    epo= 200
    batch_size = 256

    #load loss information
    best3 = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best3.npy")
    bestlat = np.load("/data/humangen_kircherlab/Autoencoder_TM/loss_big/best_latent.npy")

    best = np.union1d(best3,bestlat)

    f, (ax1, ax2) = plt.subplots(2, 1)

    #refrences 
    modl2 = LogisticRegression(random_state=2020, solver='lbfgs',max_iter=20, penalty="l2").fit(train,Y_train)

    y_pred = modl2.predict(mat_test)
    y_pred_proba = modl2.predict_proba(mat_test)[::,1]

    fpr1, tpr1, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print(auc)
    auc1 = np.round(auc, decimals=4)
    del modl2
  


    for i in best:
        layer = re.search(r'layer_(\d+)', i).group(1)
        lat =int(re.search(r'latent_(\d+)', i).group(1))
      

    
        autoencoder = keras.models.load_model("models_gross/"+str(i),custom_objects={"sparse_mse" : fc.sparse_mse},)

        

        #extracting the latent space
        latent_label_1 = autoencoder.encoder.predict(X_train_1, verbose=0)
        latent_label_0 = autoencoder.encoder.predict(X_train_0, verbose=0)
        latent_test_0 = autoencoder.encoder.predict(test_0, verbose=0)   
        latent_test_1 = autoencoder.encoder.predict(test_1, verbose=0)

            
        np.save(file="latent/"+str(i)+"__best1",arr=latent_label_1)
        np.save(file="latent/"+str(i)+"__best0",arr=latent_label_0)
        np.save(file="latent/"+str(i)+"__best_test0",arr=latent_test_0)
        np.save(file="latent/"+str(i)+"__best_test1",arr=latent_test_1)
        #np.save(file="latent/layer_"+str(layer)+"__latent_"+str(lat)+"__best",arr=latent_test)


        # #getting labels in training set
        (x1,y1) = latent_label_1.shape
        (x0,y0) =latent_label_0.shape

        labels_train = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
        latent_space_train = np.vstack((latent_label_1,latent_label_0))

        (x1,y1) = latent_test_1.shape
        (x0,y0) = latent_test_0.shape

        labels_test = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
        latent_space_test = np.vstack((latent_test_1,latent_test_0))

        modl = LogisticRegression(random_state=2020, solver='lbfgs',max_iter=20, penalty="l2").fit(latent_space_train,labels_train)

        y_pred = modl.predict(latent_space_test)
        y_pred_proba = modl.predict_proba(latent_space_test)[::,1]

        fpr, tpr, _ = metrics.roc_curve(labels_test,  y_pred_proba)
        auc = metrics.roc_auc_score(labels_test, y_pred_proba)
        print(i)
        print(auc)
        auc2 = np.round(auc, decimals=4)

        
        ax1.plot(fpr,tpr,label="latentdim "+str(lat)+" layer "+str(layer)+" AUC="+str(auc2))
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        f.savefig("figures/roc_best_all.pdf", bbox_inches='tight')
        

        if np.isin(i,best3):
            #ax2.subplot(212)
            
            ax2.plot(fpr,tpr,label="latentdim "+str(lat)+" layer "+str(layer)+" AUC="+str(auc2))
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
      
        
        


    
    

    #plt.subplot(212)
    ax1.plot(fpr1,tpr1,label="without autoencoder AUC="+str(auc1))
    ax1.set_title('best per Latent space')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax1.title.set_text('Best autoencoder per latent space')
    plt.tight_layout()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

    #plt.subplot(211)
    ax2.plot(fpr1,tpr1,label="without autoencoder AUC="+str(auc1))
    ax2.set_title('top 3')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    ax2.title.set_text('Top 3 autoencoder')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    f.savefig("figures/roc_best_all.pdf", bbox_inches='tight')
    
    
    
    
    