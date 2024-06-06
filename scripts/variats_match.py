from pickle import FALSE
import numpy as np
import pandas as pd

#filepath = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted.anno.tsv.gz"
#variants = pd.read_csv(filepath, sep='\t',low_memory=False)
#print(variants.columns)
#variants=np.load(filepath)


path0 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted.anno_Benign.tsv.gz"
path1 = "/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_Patho.anno.tsv.gz"

# Import data
variants0 = pd.read_csv(path0, sep='\t', low_memory=False)
variants1 = pd.read_csv(path1, sep='\t', low_memory=False)

variants1 = variants1.assign(labels = np.ones(variants1.shape[0]))

variants0 = variants0.assign(labels = np.zeros(variants0.shape[0]))


# Count the frequency of each variant in both data frames
df0_counts = variants0['GeneID'].value_counts()
df1_counts = variants1['GeneID'].value_counts()

#print(variants0[variants0['GeneID']=="ENSG00000000005"])
print(variants0['GeneID'].value_counts())


# Get unique variants
all_variants = set(df0_counts.index).union(set(df1_counts.index))
print(all_variants)
print(len(all_variants))



# Initialize an empty data frame to store matched variants and their frequencies
matched_variants0 = pd.DataFrame(columns=variants0.columns)
matched_variants1 = pd.DataFrame(columns=variants1.columns)

# Match the frequencies of variants in both data frames
for variant in all_variants:
    #print(variant)
    frequency_0 = df0_counts.get(variant, 0)
    frequency_1 = df1_counts.get(variant, 0)
  
    # If both frequencies are available and match, add the variant to the matched data frame
    if frequency_0 == frequency_1:
        matched_variants0 = pd.concat([matched_variants0, variants0[variants0['GeneID'] == variant]])
        matched_variants1 = pd.concat([matched_variants1, variants1[variants1['GeneID'] == variant]])
        #print(matched_variants.shape)
    elif frequency_0 <= frequency_1:
        matched_variants0 = pd.concat([matched_variants0, variants0[variants0['GeneID'] == variant]])
        matched_variants1 = pd.concat([matched_variants1, variants1[variants1['GeneID'] == variant].sample(n=frequency_0, random_state=42)]) 
        #für eins zufällige auswahl
    else:
        matched_variants0 = pd.concat([matched_variants0, variants0[variants0['GeneID'] == variant].sample(n=frequency_1, random_state=42)])
        matched_variants1 = pd.concat([matched_variants1, variants1[variants1['GeneID'] == variant]])

print(matched_variants0.columns)

print(matched_variants1[matched_variants1['GeneID']=="ENSG00000004139"])
print(matched_variants0[matched_variants0['GeneID']=="ENSG00000004139"])

vcf0 = pd.DataFrame({'CHROM' :  matched_variants0['#Chrom'], "POS":matched_variants0['Pos'], "ID":matched_variants0['GeneID'] ,"REF": matched_variants0['Ref'],"ALT":matched_variants0['Alt']})
vcf1 = pd.DataFrame({'CHROM' :  matched_variants1['#Chrom'], "POS":matched_variants1['Pos'], "ID":matched_variants1['GeneID'] ,"REF": matched_variants1['Ref'],"ALT":matched_variants1['Alt']})

vcf0.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_benign.vcf', sep='\t', index=False,header=False)
vcf1.to_csv('/data/humangen_kircherlab/Autoencoder_TM/imputed_datasets/variantsSorted_patho.vcf', sep='\t', index=False,header=False)

