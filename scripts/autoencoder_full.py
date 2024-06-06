from tokenize import Double
import numpy as np
from scipy.sparse import load_npz
import tensorflow  as tf
import tensorflow.keras

from tensorflow.keras import regularizers



from numpy.random import seed
from sklearn.model_selection import train_test_split

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import click

import functions as fc


#by running the script give latent space dimension with --latent x
@click.command()    
@click.option(
    "--latent",
    "latent",
    required=True,
    multiple=True,
    type=float,
    default=[0.5, 0.75],
    help="dimensions of latent space dependent on input dimensions , defalt 0.5 , 0.75",
)
def cli(latent):
    seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    #check if GPU is avaiable
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # path to data 
    trainFile='/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/train_v1-7-GRCh38.npz' 
    trainFile2 = '/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/full_dataset/test_v1-7-GRCh38.npz' 

    dense=False
    #toDense True if output ist dense, False if output ist sparse
    toDense=False

    #import data mat_train is data and y_train are labels(0=benign,1=pathogenic)
    X_train, y_train = fc.load_training_data(dense, toDense, trainFile, True)
    X_test, y_test = fc.load_training_data(dense, toDense, trainFile, True)

    #deleting unused data
    del y_test
    del y_train

    # convert scipy sparse matrix to sparse tensor
    X_train = fc.convert_sparse_matrix_to_sparse_tensor(X_train)
    X_test =fc.convert_sparse_matrix_to_sparse_tensor(X_test) 
 
    #input of autoencoder -> used for optimazation
    shap = X_train.shape[1:]
    #latent_dim = np.round(np.array(shap)*(0.0625,0.125,0.25,0.5,0.75,1.25,1.5,1.75,2,8,16,32,50,64,128)).astype(int)
    latent_dim = np.round(np.array(shap)*(latent)).astype(int)
    #if trained autoencoder are used for shap
    #activation_en='relu'
    #activation_de='relu'
    #hyperparameter decided on first hyperparameter optimization step
    activation_en='leaky_relu'
    activation_de='leaky_relu'
    droprate=0.1
    layer=[1,2]
    epo= 200
    batch_size = 256
    

    #initialize liste in which the loss information is saved
    loss_liste = []    
    label_loss_liste=[]

    #optimazation over latent space
    for lat in latent_dim:
        #calculationg size of second layer in decoder and encoder
        layer2_dim = round((abs(shap[0] - lat )) / 2)

        for lay in layer:
        
            #clearing moodels from memory
            tf.keras.backend.clear_session()

            #get label for trained autoencoder
            labeling = "layer_"+str(lay)+"__latent_"+str(lat)+"feature"
        
            #defining early stopping
            callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8),
            tf.keras.callbacks.CSVLogger("loss_big/"+str(labeling), separator=",", append=False)]

            autoencoder=[]                        
            #defining the autoencoder 
            if lay == 2:
                autoencoder = fc.Autoencoder2(latent_dim=lat, layer2_dim= layer2_dim, shape=shap,activation_en=activation_en,activation_de= activation_de,droprate=droprate)
            else:
                autoencoder = fc.Autoencoder1(latent_dim=lat, layer2_dim= layer2_dim, shape=shap,activation_en=activation_en,activation_de= activation_de,droprate=droprate)
                
            
                
            autoencoder.compile(optimizer='adam', loss=fc.sparse_mse)

            #fitting the autoencoder with the data
            autoencoder.fit(X_train,X_train,epochs=epo,callbacks=[callback], batch_size=batch_size, shuffle=True,validation_data=(X_test,X_test),verbose=2)

            #appending loss information
            label_loss_liste.append(labeling)
            loss_liste.append( autoencoder.evaluate (X_train,X_train,batch_size=batch_size))
            
            #save autoencoder
            autoencoder.save(filepath='models_gross/'+str(labeling))
       
        
        #safe loss of models
        np.save(file="loss_big/all_loss__"+str(labeling),arr=loss_liste)    
        np.save(file="loss_big/all_loss_label__"+str(labeling),arr=label_loss_liste)   



if __name__ == "__main__":
  cli()




