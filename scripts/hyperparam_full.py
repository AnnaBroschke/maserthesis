import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
from matplotlib.colors import LogNorm
from numpy.random import seed


if __name__ == "__main__":
    seed(42)
    np.random.seed(42)
   
    #latent dimension sizes
    #latent_dim = np.round(np.array(1213)*(0.0625,0.125,0.25,0.5,0.75,1.25,1.5,1.75,2)).astype(int) #12
    latent_dim = np.round(np.array(1213)*(0.0625,0.125,0.25,0.5,0.75, 1.25)).astype(int)
    
    #hyperparameter options
    layer = [1,2]
    #choosen in first hyperparameter search
    activation_en='leaky_relu'
    activation_de='leaky_relu'
    droprate=0.1
            
    #starting point of array in which all losses get appended
    all_loss =np.zeros(2)
    all_loss_label = np.zeros(2)
    
    #for loop over all hyperparameter (latent space and layers)
    for lat in latent_dim:
        hilf = []
        hilf_label =[]
        
        for lay in layer:
            #get label of autoencoder
            labeling = "layer_"+str(lay)+"__latent_"+str(lat)+"gross_full"
            loss = np.loadtxt("/data/humangen_kircherlab/Autoencoder_TM/loss_big/"+str(labeling),skiprows=1,delimiter=",",usecols = (-1))[-1]

            #get loss to append in helper varible which is added to the total loss (all_loss) array
            hilf.append(loss)
            hilf_label.append(labeling)
            
            
        all_loss = np.vstack((all_loss,hilf))
        all_loss_label = np.vstack((all_loss_label,hilf_label))
        # in tabelle rein tun
         # löschen von hilfsvariable

    #deleting zeros in the first row
    all_loss = all_loss[1:,:]
    all_loss_label = all_loss_label[1:,:]

    label_flat = all_loss_label.flatten()
    
    #identify best 3 autoencoder according to loss
    mins = label_flat[np.argsort(all_loss.flatten())[:3]]

    #print out best autoencoders
    print("top 3 best autoencoders")
    print(mins)
    shap = all_loss.shape[0]
    
    #go through all latent spaces to get best autoencoder per latent space
    print("best autoencoders per latent space ")
    latbest = ['a'] * shap
    for i in range(shap):
        label_i = all_loss_label[i,:]
        latbest[i] = label_i[np.argmin(all_loss[i,])]

        
        print(latbest[i])

    
    #transfer np array to pd dataframe to print out heatmap
    df_loss= pd.DataFrame(all_loss, index=latent_dim, columns= [1,2])

    np.save(file="loss_big/best3_full",arr=mins)    
    np.save(file="loss_big/best_latent_full",arr=latbest)   

       

    sns.heatmap(df_loss,annot=True,norm=LogNorm())
    plt.xlabel('layers')
    plt.ylabel('latent space dimensions')
    plt.savefig("figures/loss_gross_full.pdf")

    

    #sns.heatmap(df_loss,annot=False, vmax=20)
    #plt.xlabel('hyperparameters')
    #plt.ylabel('latent space dimensions')
    #plt.savefig("figures/loss_20.png")
            

    #print(all_loss)
    #print(labels_loss)



    
