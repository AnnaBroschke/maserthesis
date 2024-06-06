# Read me

In this folder are all the scripts I wrote in my master thesis. All scripts are listed below with the task what they accomplish. The order in which they are listed should be the order in which they should be executed.

## functions.py
In this script the used functions are saved. Do to:
- load in test and training data
- definition of the autoencoder model (for both one and two dense layers)
- konvert scipy sparse array in sparse tensor
- custom mse function to calculate mse for each batch

## dataset.py
In this script, important values of the data set are calculated in order to get an overview of the type of data the following scripts work with.
### Needed data
- training data

## autoencoder.py
In this script all the autoencoders are trained for the first hyperparameter optimization
### Needed data
- training data

## hyperparam.py
This script plots the heatmap of the first hyperparameter optimization on the big grid.
### Needed data
- trained autoencoders

## autoen_gross.py
In this script the autoencoders on the training data without feature crosses are trained and saved. To run this script the latent space needs to be clarified in a relative way.
### Needed data
- traing data without cross features
- size of latent space

## hyperparam_gross.py
This script plots the heatmap of losses of autoencoder without cross features and produces list of loss info with best autoencoders per latent space.
### Needed data
- trained autoencoders without cross features

## autoencoder_full.py
In this script the autoencoders on the full feature space are trained and saved. To run this script the latent space needs to be clarified in a relative way.
### Needed data
- traing data with full feature space
- size of latent space

## hyperparam_full.py
This script plots the heatmap of losses of autoencoder with full feature space and produces list of loss info with best autoencoders per latent space.
### Needed data
- trained autoencoders with full feature space

## logisticsave.py
This script traines and saves the logistic regression modells like CADD with sklkearn.
### Needed data
- training data
- trained autoencoder
- loss information

## matched_variant_neu.py
This script matches up the variants to the gene in the *All Variants (Clinvar vs.ExAC)* dataset to get variants for dataset *matched Variants* and *Downsapmpled Matched Variants*
### Needed data
- tsv file of variants in *All Variants (Clinvar vs.ExAC)* set

## barplot.py
This script produces a barplot figure of all AUC values derived from all tested autoencoders. 
### Needed data
- training data (both sets)
- test data (all 4 sets)
- traines autoencoder
- trained logistic regression models

## logistg_tensorflow.py
This script traines the logistic regression model inside the autoencoder for the sake of having one model for SHAP.
### Needed data
- trained autoencoder with activation function relu

## feature_auc.py
This script plots ROC curves for each test set for the logistic regression models for the feature importance with SHAP.
### Needed data
- trained logistic regression autoencoder models
- test data (all 4 sets)

## feature_importance.py
This scripts calculates the SHAP values of the logistic regression autencoder model. It also shows the SHAP values in a plot and a latex table output.
### Needed data
- train data
- test data
- traines logistig regression autoencoder model


