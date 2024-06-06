import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
from matplotlib.colors import LogNorm


if __name__ == "__main__":

    # liste latent space sizes
    
    #461 is the number of features in the dataset without cross-features
    #change 461 into 1213 to fit the skript to the full feature space
    latent_dim = np.round(np.array(461)*(0.0625,0.125,0.25,0.5,0.75,1.25,1.5,1.75,2,8,16,32)).astype(int) #12
    #print(latent_dim)
    
    #used other hyperparameter
    layer = [1,2]
    activation_en=['tanh','sigmoid','leaky_relu']
    activation_de=['tanh','sigmoid','leaky_relu']
    droprate = [0.1,0.2,0.4]
    

        
    #to store the loss of each model and the label of the models
    all_loss =np.zeros(54)
    all_loss_label = np.zeros(54)
    #if the used model is with the full feature space (only 36 models werew trained)
    #all_loss =np.zeros(36)
    #all_loss_label = np.zeros(36)

    #extract for every possiple hyperparametercombination the loss information
    for lat in latent_dim:
        #helper varibles to store the loss and labels
        hilf = []
        hilf_label =[]
        for drop in droprate:
            for lay in layer:
                for ac_en in activation_en:
                    for ac_de in activation_de:
                        #get labeling
                        labeling = "layer_"+str(lay)+"__latent_"+str(lat)+"__activation_encoder_"+str(ac_en)+"__activation_decoder_"+str(ac_de)+"__droprate_"+str(drop)
                        #get loss
                        loss = np.loadtxt("/data/humangen_kircherlab/Autoencoder_TM/models/"+labeling,skiprows=1,delimiter=",",usecols = (-1))[-1]
                        
                        
                        hilf.append(loss)
                        hilf_label.append(labeling)
        #add losses to table of all labels and losses           
        all_loss = np.vstack((all_loss,hilf))
        all_loss_label = np.vstack((all_loss_label,hilf_label))
        
    #delete zeros
    all_loss = all_loss[1:,:]
    all_loss_label = all_loss_label[1:,:]
    label_flat = all_loss_label.flatten()
    

    #extract 3 minimum values
    mins = label_flat[np.argsort(all_loss.flatten())[:3]]
    print("top 3 best autoencoders")
    print(mins)
    shap = all_loss.shape[0]
    
    #extract best autoencoder per latent space
    print("best autoencoders per latent space ")
    for i in range(shap):
        label_i = all_loss_label[i,:]
        label_min = label_i[np.argmin(all_loss[i,])]
        print(label_min)

    
    #to show heatmap transformation to pandas dataframe
    df_loss= pd.DataFrame(all_loss, index=latent_dim)

    #heatmap of all losses with a logarithmic scale

    #sns.heatmap(df_loss,annot=False,norm=LogNorm())
    #plt.xlabel('hyperparameter combination ID')
    #plt.ylabel('latent space dimensions')
    #plt.savefig("figures/loss.svg")

    
    #heatmap with maximum of 100 to diffrenciate losses with small losses
    sns.heatmap(df_loss,annot=False, vmax=100)
    plt.xlabel('hyperparameter combination ID')
    plt.ylabel('latent space dimensions')
    plt.savefig("figures/loss_100.svg")
            





    
