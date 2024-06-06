from pickle import FALSE
import numpy as np
import pandas as pd



#import variants in ClinVar vs ExAc 
path0 ="/data/humangen_kircherlab/CADD1_7_GRCh38_TM/data/GRCh38/ExAcBenign/annotation/variantsSorted.anno.tsv.gz"
path1 ="/data/humangen_kircherlab/CADD1_7_GRCh38_TM/data/GRCh38/clinVarPatho/annotation/variantsSorted.anno.tsv.gz"
# Import data
variants0 = pd.read_csv(path0, sep='\t', low_memory=False)
variants1 = pd.read_csv(path1, sep='\t', low_memory=False)

#assign labels
variants1 = variants1.assign(labels = np.ones(variants1.shape[0]))
variants0 = variants0.assign(labels = np.zeros(variants0.shape[0]))

#merge dataframes
df_all_genes = pd.concat([variants0,variants1])
#print(df_all_genes.shape)

#get unique values of gene ID
df_all_genes = df_all_genes.drop_duplicates(subset=["GeneID"])["GeneID"]
#print(df_all_genes.shape)
list_all_genes = df_all_genes.to_list()



# Initialize an empty data frame to store matched variants and their frequencies
matched_variants0 = pd.DataFrame(columns=variants0.columns)
matched_variants1 = pd.DataFrame(columns=variants1.columns)

# Match the frequencies of variants in both data frames
for gene in list_all_genes:
    
    #subset of all variants from with gene ID
    gene_0 = variants0[variants0['GeneID'] == gene]
    gene_1 = variants1[variants1['GeneID'] == gene]


    # If both frequencies are available and match, add the variant to the matched data frame
    if gene_0.shape[0] == gene_1.shape[0]:
        #for case of downsampled matched variants add maxima of 5
        if gene_0.shape[0] >= 5:
            matched_variants0 = pd.concat([matched_variants0, gene_0.sample(n=5, random_state=42)])
            matched_variants1 = pd.concat([matched_variants1, gene_1.sample(n=5, random_state=42)])
        else:
            matched_variants0 = pd.concat([matched_variants0, gene_0])
            matched_variants1 = pd.concat([matched_variants1, gene_1])
    # cases in which no variants per gene   
    elif gene_0.shape[0] ==0:
        a=1
    elif gene_1.shape[0] ==0:
        a=1
    # when one dataset has more variants than the other the selection is random
    elif gene_0.shape[0] < gene_1.shape[0]:
        #add a maxima of five in case of downsampled matched variants
        if gene_0.shape[0] >=5:
            matched_variants0 = pd.concat([matched_variants0, gene_0.sample(n=5, random_state=42)])
            matched_variants1 = pd.concat([matched_variants1, gene_1.sample(n=5, random_state=42)])
        else:
            matched_variants0 = pd.concat([matched_variants0, gene_0])
            matched_variants1 = pd.concat([matched_variants1, gene_1.sample(n=gene_0.shape[0], random_state=42)]) 
    elif gene_0.shape[0] > gene_1.shape[0]:
        if gene_1.shape[0] >= 5:
            matched_variants0 = pd.concat([matched_variants0, gene_0.sample(n=5, random_state=42)])
            matched_variants1 = pd.concat([matched_variants1, gene_1.sample(n=5, random_state=42)])
        else:
            matched_variants0 = pd.concat([matched_variants0, gene_0.sample(n=gene_1.shape[0], random_state=42)])
            matched_variants1 = pd.concat([matched_variants1, gene_1])
    #error if no case is possiple
    else:
        print("error")

#print(matched_variants0.columns)

#get vcf files from resulting datasets
vcf0 = pd.DataFrame({'CHROM' :  matched_variants0['#Chrom'], "POS":matched_variants0['Pos'], "ID":matched_variants0['GeneID'] ,"REF": matched_variants0['Ref'],"ALT":matched_variants0['Alt']})
vcf1 = pd.DataFrame({'CHROM' :  matched_variants1['#Chrom'], "POS":matched_variants1['Pos'], "ID":matched_variants1['GeneID'] ,"REF": matched_variants1['Ref'],"ALT":matched_variants1['Alt']})

#sorted nach chrom numerisch 1,2...,X,Y und dann POS 
chrom_sort_order = {'X': float('inf')}
#chrom_sort_order.update({str(i): i for i in range(1, 23)})
vcf0 = vcf0.sort_values(["CHROM","POS"],key=lambda x: x.map(chrom_sort_order))


#save resulting vcf files
vcf0.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_benign_down.vcf', sep='\t', index=False,header=False)
vcf1.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_patho_down.vcf', sep='\t', index=False,header=False)

vcf0.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_benign_down.csv', sep='\t', index=False)
vcf1.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_patho_down.csv', sep='\t', index=False)



