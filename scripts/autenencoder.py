from tokenize import Double
import numpy as np
from scipy.sparse import load_npz
import tensorflow  as tf
import pandas as pd
import keras
from keras import layers
from keras import regularizers


import sklearn
from numpy.random import seed
from sklearn.model_selection import train_test_split

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import click






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

# vonverts a scipysparse matrix to sparse tensor
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

#Autoencoder with 2 layer design
class Autoencoder2(Model):
        def __init__(self, latent_dim,layer2_dim,activation_en,activation_de, shape,droprate):
            super(Autoencoder2, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=shape,sparse=True),
                #layers.Flatten(),
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

#Autoencoder with one layer design  
class Autoencoder1(Model):
        def __init__(self, latent_dim,activation_en,activation_de, shape,droprate):
            super(Autoencoder1, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=shape,sparse=True),
                #layers.Flatten(),
                #layers.Dropout(rate=droprate, seed=2020),
                layers.Dense(latent_dim, activation=activation_en)
                
            ])
            self.decoder = tf.keras.Sequential([
                #layers.Dense(tf.math.reduce_prod(shape), activation=activation_de,activity_regularizer=regularizers.l1(l1=regu))
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

#by running the script droprate and number of layers need to be defined
@click.command()
@click.option(
    "--droprate",
    "droprate",
    required=True,
    multiple=False,
    type=float,
    default=0.1,
    help="percentage of droped data in droping layer, defalt 0.1",
)
@click.option(
    "--layer",
    "layer",
    required=True,
    multiple=False,
    type=int,
    default=1,
    help="number of layers (1,2), defalt 1",
)
def cli(droprate,layer):
    #controll if value for droprate is false
    if 0<= droprate >=1:
        print("Wrong number for droprate, chopose a number between 0 and 1")
        

    # path to data 
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/test_v1-7-GRCh38.npz'

    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False

    #import data mat_train is data and y_train are labels(0=benign,1=pathogenic)
    mat_train, y_train = load_training_data(dense, toDense, trainFile, True)

    #splitting test and traindata for validation of the autoencoder
    train, test = train_test_split(mat_train, train_size = 0.8, random_state = seed(2020))


    # convert scipy sparse matrix to sparse tensor
    X_train = convert_sparse_matrix_to_sparse_tensor(train[:,:])
    X_test =convert_sparse_matrix_to_sparse_tensor(test[:,:]) 
 
    #input of autoencoder -> used for optimazation
    shap = X_train.shape[1:]
    latent_dim = np.round(np.array(shap)*(0.0625,0.125,0.25,0.5,0.75,1.25,1.5,1.75,2,8,16,32)).astype(int)
    activation_en=['tanh','sigmoid','leaky_relu']
    activation_de=['tanh','sigmoid','leaky_relu']
    epo= 200
    batch_size = 256


    #initialize list to store losses
    loss_liste = []    
    label_loss_liste=[]

    #optimazation over latent space
    for lat in latent_dim:
        #calculationg size of second layer in decoder and encoder
        layer2_dim = round((abs(shap[0] - lat )) / 2)
        #optimazation over the activation function
        for ac_en in activation_en:
            for ac_de in activation_de:
                
                #clearing moodels from memory
                tf.keras.backend.clear_session()

                #labeling the trained autoencoder with used hyperparameter
                labeling = "layer_"+str(layer)+"__latent_"+str(lat)+"__activation_encoder_"+str(ac_en)+"__activation_decoder_"+str(ac_de)+"__droprate_"+str(droprate)
            
                #defining early stopping
                callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8),
                keras.callbacks.CSVLogger("models/"+str(labeling), separator=",", append=False)]

                autoencoder=[]                        
                #defining the autoencoder 
                if layer == 2:
                    autoencoder = Autoencoder2(latent_dim=lat,layer2_dim= layer2_dim, shape=shap,activation_en=ac_en,activation_de= ac_de,droprate=droprate)
                elif layer==1:
                    autoencoder = Autoencoder1(latent_dim=lat, shape=shap,activation_en=ac_en,activation_de= ac_de,droprate=droprate)
                    
                else:
                    #controll if value for layer is false
                    print("Wrong value in layer, choose eather 1 or 2")
                    
                autoencoder.compile(optimizer='adam', loss=sparse_mse)

                #fitting the autoencoder with the data
                autoencoder.fit(X_train,X_train,epochs=epo,callbacks=[callback], batch_size=batch_size, shuffle=False,validation_data=(X_test,X_test))

                #appending the loss information
                label_loss_liste.append(labeling)
                loss_liste.append( autoencoder.evaluate (X_train,X_train,batch_size=batch_size))

                #extracting the latent space
                latent = autoencoder.encoder.predict(X_train)
                

                print(labeling)
                

                #saving the model and the latent space
                autoencoder.save("models/layer_2_latent"+str(lat)+"activation_encoder"+str(ac_en)+"activation_decoder"+str(ac_de))
                np.save(file="latent/"+str(labeling),arr=latent)

        #save loss information
        np.save(file="models/all_loss__latent_"+str(lat)+"__droprate_"+str(droprate),arr=loss_liste)    
        np.save(file="models/all_loss_label__latent_"+str(lat)+"__droprate_"+str(droprate),arr=label_loss_liste)   



if __name__ == "__main__":
  cli()
