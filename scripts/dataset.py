import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import scipy.sparse as sc




#methods taken from https://github.com/kircherlab/CADD/blob/master/scripts/sklearnTools/handling.py
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
        print("ich fuehre das hier aus obwohl ich das nicht soll")
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



if __name__ == "__main__":
    # path to data 
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz'
    trainFile2='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/train_v1-7-GRCh38.npz'
    dense=False
    toDense=False

    #import data mat_train is data and y_train are labels(0=benign,1=pathogenic)
    mat_train, y_train = load_training_data(dense, toDense, trainFile, True)
    mat_train2, y_train2 = load_training_data(dense, toDense, trainFile2, True)

    mat_test = sc.vstack((mat_train,mat_train2))
    y_train = np.expand_dims(y_train, axis=1)
    y_train2 = np.expand_dims(y_train2, axis=1)
    y_train =np.vstack((y_train,y_train2))



    #print out important values
    print('shape of trainigset '+ str(mat_train.shape))
    print('shape of labels ' + str(y_train.shape))
    print('is.nan in trainingset '+str(np.isnan(mat_train.sum())))
    print('is.nan in labels '+str(np.isnan(y_train.sum())))
    print('min value of training '+str(np.min(mat_train)))
    print('max value of training '+str(np.max(mat_train)))
    print('min value of labels '+str(np.min(y_train)))
    print('max value of labels '+str(np.max(y_train)))
    print('number 1 values in label '+str(sum(y_train==1)))
    print('number 0 values in label '+str(sum(y_train==0)))


