from turtle import shape
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
import umap
import umap.plot

#methods taken from https://github.com/kircherlab/CADD/blob/master/scripts/sklearnTools/handling.py


if __name__ == "__main__":
    # path to data 
    #load latent space according to the label
    latent_label_1 = np.load("/data/humangen_kircherlab/Autoencoder_TM/latent/best__label1.npy")
    latent_label_0 = np.load("/data/humangen_kircherlab/Autoencoder_TM/latent/best__label0.npy")
    latent_test_1 = np.load("/data/humangen_kircherlab/Autoencoder_TM/latent/best__test1.npy")
    latent_test_0 = np.load("/data/humangen_kircherlab/Autoencoder_TM/latent/best__test0.npy")
    

    
    (x1,y1) = latent_label_1.shape
    (x0,y0) =latent_label_0.shape

    #stack them togeter inclusive label
    labels_train = np.ravel(np.vstack((np.zeros((x1,1)),np.ones((x0,1)))))
    latent_space_train = np.vstack((latent_label_1,latent_label_0))

    #train k-means algorithm to cluster benign and pathogenic
    kmeans = KMeans(n_clusters=2, random_state=2020, n_init="auto").fit(latent_space_train)
    
    #print(kmeans.labels_.shape)
    #112 von 5000 samples
    #get table how good clustered label are in consent to real labels
    vierfeld=np.ones((2,2))
    vierfeld[0,0] = sum(np.logical_and(kmeans.labels_ == 0, labels_train == 0))
    vierfeld[1,1] = sum(np.logical_and(kmeans.labels_ == 1, labels_train == 1))
    vierfeld[0,1] = sum(np.logical_and(kmeans.labels_ == 0, labels_train == 1))
    vierfeld[1,0] = sum(np.logical_and(kmeans.labels_ == 1, labels_train == 0))
    print("trainings data")
    print(vierfeld)
   
    print("Von k-means gelabelt als 1 sind"+str(100*(vierfeld[1,1]/(vierfeld[1,1]+vierfeld[1,0])))+"prozent 1 gelabelt")
    print("Von k-means gelabelt als 0 sind"+str(100*(vierfeld[0,1]/(vierfeld[0,1]+vierfeld[0,0])))+"prozent 1 gelabelt")

    labels_train =labels_train.reshape((latent_space_train.shape[0],1))

    #print(latent_space_train.shape)
    #print(labels_train.shape)

    latent_space_train = np.hstack((latent_space_train,labels_train))

    #print out UMAP of latent space with real labels
    mapper = umap.UMAP().fit(latent_space_train[:,:-1])
    umap.plot.points(mapper, labels=latent_space_train[:,-1])
    plt.savefig("figures/umap_best_train.png")


    