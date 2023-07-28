# LR multi-tissue cross-comparison

##### Ver:: A2_V6
##### Author(s) : Issac Goh
##### Date : 220823;YYMMDD
### Author notes
#    - Current defaults scrpae data from web, so leave as default and run
#    - slices model and anndata to same feature shape, scales anndata object
#    - added some simple benchmarking
#    - creates dynamic cutoffs for probability score (x*sd of mean) in place of more memory intensive confidence scoring
#    - Does not have majority voting set on as default, but module does exist
#    - Multinomial logistic relies on the (not always realistic) assumption of independence of irrelevant alternatives whereas a series of binary logistic predictions does not. collinearity is assumed to be relatively low, as it becomes difficult to differentiate between the impact of several variables if this is not the case
#    - Feel free to feed this model latent representations which capture non-linear relationships, the model will attempt to resolve any linearly seperable features. Feature engineering can be applied here.
    
### Features to add
#    - Add ability to consume anndata zar format for sequential learning
### Modes to run in
#    - Run in training mode
#    - Run in projection mode
#libraries

# import pkg_resources
# required = {'harmonypy','sklearn','scanpy','pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn' ,'scipy'}
# installed = {pkg.key for pkg in pkg_resources.working_set}
# missing = required - installed
# if missing:
#    print("Installing missing packages:" )
#    print(missing)
#    python = sys.executable
#    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
import sys
import subprocess
from collections import Counter
from collections import defaultdict
import scanpy as sc
import pandas as pd
import pickle as pkl
import numpy as np
import scipy
import matplotlib.pyplot as plt
import re
import glob
import os
import sys
#from geosketch import gs
from numpy import cov
import scipy.cluster.hierarchy as spc
import seaborn as sns; sns.set(color_codes=True)
from sklearn.linear_model import LogisticRegression
import sklearn
from pathlib import Path
import requests
import psutil
import random
import threading
import tracemalloc
import itertools
import math
import warnings
import sklearn.metrics as metrics
from collections import Counter
from collections import defaultdict
import scanpy as sc
import pandas as pd
import pickle as pkl
import numpy as np
import scipy
import matplotlib.pyplot as plt
import re
import glob
import os
import sys
#from geosketch import gs
from numpy import cov
import scipy.cluster.hierarchy as spc
import seaborn as sns; sns.set(color_codes=True)
from sklearn.linear_model import LogisticRegression
import sklearn
from pathlib import Path
import requests
import psutil
import random
import threading
import tracemalloc
import itertools
import math
import warnings
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import log_loss
import mygene
import gseapy as gp
import mygene
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.sparse import csr_matrix
from scipy.stats import entropy
# Main_plotting_modules

def plot_label_probability_heatmap(pred_out):
    """
    General description: 
    Plots a heatmap of the median probabilities of each label for each original label.
    
    Parameters:
    pred_out (pd.DataFrame): A dataframe where each row corresponds to a cell, each column corresponds to a label, and each entry is the probability of the cell being of the label. The dataframe should also contain a 'predicted' column with the predicted label of each cell and an 'orig_labels' column with the original label of each cell.
    """
    model_mean_probs = pred_out.loc[:, pred_out.columns != 'predicted'].groupby('orig_labels').median()
    model_mean_probs = model_mean_probs * 100
    model_mean_probs = model_mean_probs.dropna(axis=0, how='any', subset=None, inplace=False)
    crs_tbl = model_mean_probs.copy()

    # Sort dataframe columns by rows
    index_order = list(crs_tbl.max(axis=1).sort_values(ascending=False).index)
    col_order = list(crs_tbl.max(axis=0).sort_values(ascending=False).index)
    crs_tbl = crs_tbl.loc[index_order]
    crs_tbl = crs_tbl[col_order]

    plt.figure(figsize=(20,15))
    sns.set(font_scale=0.5)
    g = sns.heatmap(crs_tbl, cmap='viridis_r',  annot=False, vmin=0, vmax=max(np.max(crs_tbl)), linewidths=1, center=max(np.max(crs_tbl))/2, square=True, cbar_kws={"shrink": 0.5})

    plt.ylabel("Original labels")
    plt.xlabel("Training labels")
    plt.show()

def plot_crosstab_heatmap(adata, x, y):
    """
    General description.
    Plots a heatmap of the cross-tabulation of two categorical attributes.

    Parameters:
    adata (anndata.AnnData): The annotated data matrix.
    x (str): The name of the attribute to use for the rows of the cross-tabulation.
    y (str): The name of the attribute to use for the columns of the cross-tabulation.
    """
    # Compute the cross-tabulation
    x_attr = adata.obs[x]
    y_attr = adata.obs[y]
    crs = pd.crosstab(x_attr, y_attr)

    # Normalize each column to sum to 100
    crs = crs.div(crs.sum(axis=0), axis=1).multiply(100).round(2)

    # Sort the rows and columns of the cross-tabulation
    index_order = list(crs.max(axis=1).sort_values(ascending=False).index)
    col_order = list(crs.max(axis=0).sort_values(ascending=False).index)
    crs = crs.loc[index_order, col_order]

    # Plot the heatmap
    plt.figure(figsize=(20,15))
    sns.set(font_scale=0.8)
    g = sns.heatmap(crs, cmap='viridis_r', vmin=0, vmax=100, linewidths=1, center=50, square=True, cbar_kws={"shrink": 0.3})
    plt.ylabel("Original labels")
    plt.xlabel("Predicted labels")
    plt.show()

# ENSDB-HGNC Option 1 
#from gseapy.parser import Biomart
#bm = Biomart()
## view validated marts#
#marts = bm.get_marts()
## view validated dataset
#datasets = bm.get_datasets(mart='ENSEMBL_MART_ENSEMBL')
## view validated attributes
#attrs = bm.get_attributes(dataset='hsapiens_gene_ensembl')
## view validated filters
#filters = bm.get_filters(dataset='hsapiens_gene_ensembl')
## query results
#queries = ['ENSG00000125285','ENSG00000182968'] # need to be a python list
#results = bm.query(dataset='hsapiens_gene_ensembl',
#                       attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene_id', 'go_id'],
#                       filters={'ensemble_gene_id': queries}
                      
# ENSDB-HGNC Option 2
def convert_hgnc(input_gene_list):
    """
    General description.

    Parameters:

    Returns:

    """
    import mygene
    mg = mygene.MyGeneInfo()
    geneList = input_gene_list 
    geneSyms = mg.querymany(geneList , scopes='ensembl.gene', fields='symbol', species='human')
    return(pd.DataFrame(geneSyms))
# Example use: convert_hgnc(['ENSG00000148795', 'ENSG00000165359', 'ENSG00000150676'])

# Scanpy_degs_to_long_format
def convert_scanpy_degs(input_dataframe):
    """
    General description.

    Parameters:

    Returns:

    """
    if 'concat' in locals() or 'concat' in globals():
        del(concat)
    degs = input_dataframe
    n = degs.loc[:, degs.columns.str.endswith("_n")]
    n = pd.melt(n)
    p = degs.loc[:, degs.columns.str.endswith("_p")]
    p = pd.melt(p)
    l = degs.loc[:, degs.columns.str.endswith("_l")]
    l = pd.melt(l)
    n = n.replace(regex=r'_n', value='')
    n = n.rename(columns={"variable": "cluster", "value": "gene"})
    p = (p.drop(["variable"],axis = 1)).rename(columns={ "value": "p_val"})
    l = (l.drop(["variable"],axis = 1)).rename(columns={ "value": "logfc"})
    return(pd.concat([n,p,l],axis=1))
#Usage: convert_scanpy_degs(scanpy_degs_file)

# Clean convert gene list to list
def as_gene_list(input_df,gene_col):
    """
    General description.

    Parameters:

    Returns:

    """
    gene_list = input_df[gene_col]
    glist = gene_list.squeeze().str.strip().tolist()
    return(glist)

# No ranking enrichr
def enrich_no_rank(input_gene_list,database,species="Human",description="enr_no_rank",outdir = "./enr",cutoff=0.5):
    """
    General description.

    Parameters:

    Returns:

    """
    # list, dataframe, series inputs are supported
    enr = gp.enrichr(gene_list=input_gene_list,
                     gene_sets=database,
                     organism=species, # don't forget to set organism to the one you desired! e.g. Yeast
                     #description=description,
                     outdir=outdir,
                     # no_plot=True,
                     cutoff=cutoff # test dataset, use lower value from range(0,1)
                    )
    return(enr)
    #Usge: enrich_no_rank(gene_as_list)
    
# Custom genelist test #input long format degs or dictionary of DEGS
def custom_local_GO_enrichment(input_gene_list,input_gmt,input_gmt_key_col,input_gmt_values,description="local_go",Background='hsapiens_gene_ensembl',Cutoff=0.5):
    """
    General description.

    Parameters:

    Returns:

    """
    
    #Check if GMT input is a dictionary or long-format input
    if isinstance(input_gmt, dict):
        print("input gmt is a dictionary, proceeding")
        dic = input_gmt
    else:
        print("input gmt is not a dictionary, if is pandas df,please ensure it is long-format proceeding to convert to dictionary")
        dic =  input_gmt.groupby([input_gmt_key_col])[input_gmt_values].agg(lambda grp: list(grp)).to_dict()
        
    enr_local = gp.enrichr(gene_list=input_gene_list,
                 description=description,
                 gene_sets=dic,
                 background=Background, # or the number of genes, e.g 20000
                 cutoff=Cutoff, # only used for testing.
                 verbose=True)
    return(enr_local)
    #Example_usage: custom_local_GO_enrichment(input_gene_list,input_gmt,input_gmt_key_col,input_gmt_values) #input gmt can be long-format genes and ontology name or can be dictionary of the same   

    
# Pre-ranked GS enrichment
def pre_ranked_enr(input_gene_list,gene_and_ranking_columns,database ='GO_Biological_Process_2018',permutation_num = 1000, outdir = "./enr_ranked",cutoff=0.25,min_s=5,max_s=1000):
    """
    General description.

    Parameters:

    Returns:

    """
    glist = input_gene_list[gene_and_ranking_columns]
    pre_res = gp.prerank(glist, gene_sets=database,
                     threads=4,
                     permutation_num=permutation_num, # reduce number to speed up testing
                     outdir=outdir,
                     seed=6,
                     min_size=min_s,
                     max_size=max_s)
    return(pre_res)
    #Example usage: pre_ranked_hyper_geom(DE, gene_and_ranking_columns = ["gene","logfc"],database=['KEGG_2016','GO_Biological_Process_2018'])

    
# GSEA module for permutation test of differentially regulated genes
# gene set enrichment analysis (GSEA), which scores ranked genes list (usually based on fold changes) and computes permutation test to check if a particular gene set is more present in the Up-regulated genes, 
# among the DOWN_regulated genes or not differentially regulated.
#NES = normalised enrichment scores accounting for geneset size
def permutation_ranked_enr(input_DE,cluster_1,cluster_2,input_DE_clust_col,input_ranking_col ,input_gene_col ,database = "GO_Biological_Process_2018"):
    """
    General description.

    Parameters:

    Returns:

    """
    input_DE = input_DE[input_DE[input_DE_clust_col].isin([cluster_1,cluster_2])]
    #Make set2 negative values to represent opposing condition
    input_DE[input_ranking_col].loc[input_DE[input_DE_clust_col].isin([cluster_2])] = -input_DE[input_ranking_col].loc[input_DE[input_DE_clust_col].isin([cluster_2])]
    enr_perm = pre_ranked_enr(input_DE,[input_gene_col,input_ranking_col],database,permutation_num = 100, outdir = "./enr_ranked_perm",cutoff=0.5)
    return(enr_perm)
    #Example usage:permutation_ranked_enr(input_DE = DE, cluster_1 = "BM",cluster_2 = "YS",input_DE_clust_col = "cluster",input_ranking_col = "logfc",input_gene_col = "gene",database = "GO_Biological_Process_2018")
    #input long-format list of genes and with a class for permutaion, the logfc ranking should have been derived at the same time


#Creating similarity matrix from nested gene lists
def create_sim_matrix_from_enr(input_df,nested_gene_column="Genes",seperator=";",term_col="Term"):
    """
    General description.

    Parameters:

    Returns:

    """
#    input_df[gene_col] = input_df[gene_col].astype(str)
#    input_df[gene_col] = input_df[gene_col].str.split(";")
#    uni_val = list(input_df.index.unique())
#    sim_mat = pd.DataFrame(index=uni_val, columns=uni_val)
#    exploded_df = input_df.explode(gene_col)
#    # Ugly loop for cosine gs similarity matrix (0-1)
#    for i in (uni_val):
#        row = exploded_df[exploded_df.index.isin([i])]
#        for z in (uni_val):
#            col = exploded_df[exploded_df.index.isin([z])]
#            col_ls = col[gene_col]
#            row_ls = row[gene_col]
#            sim = len(set(col_ls) & set(row_ls)) / float(len(set(col_ls) | set(row_ls)))
#            sim_mat.loc[i , z] = sim

#    Check term col in columns else, check index as it\s sometimes heree
    if not term_col in list(input_df.columns):
        input_df[term_col] = input_df.index

#    Create a similarity matrix by cosine similarity
    input_df = input_df.copy()
    gene_col = nested_gene_column #"ledge_genes"
    input_df[gene_col] = input_df[gene_col].astype(str)
    input_df[gene_col] = input_df[gene_col].str.split(seperator)
    term_vals = list(input_df[term_col].unique())
    uni_val = list(input_df[term_col].unique())
    sim_mat = pd.DataFrame(index=uni_val, columns=uni_val)
    exploded_df = input_df.explode(gene_col)
    arr = np.array(input_df[gene_col])
    vals = list(exploded_df[nested_gene_column])
    import scipy.sparse as sparse
    def binarise(sets, full_set):
        """
        General description.

        Parameters:

        Returns: sparse binary matrix of given sets.

        """
        return sparse.csr_matrix([[x in s for x in full_set] for s in sets])
    # Turn the matrix into a sparse boleen matrix of binarised values
    sparse_matrix = binarise(arr, vals)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sparse_matrix)
    sim_mat = pd.DataFrame(similarities)
    sim_mat.index = uni_val
    sim_mat.columns = uni_val
    return(sim_mat)
#Example usage : sim_mat = create_sim_matrix_from_enr(enr.res2d)


#Creating similarity matrix from GO terms
def create_sim_matrix_from_term(input_df,nested_gene_column="Term",seperator=" ",term_col="Term"):
    """
    General description.

    Parameters:

    Returns:

    """

#    Check term col in columns else, check index as it\s sometimes heree
    if not term_col in list(input_df.columns):
        input_df[term_col] = input_df.index

#    Create a similarity matrix by cosine similarity
    input_df = input_df.copy()
    gene_col = nested_gene_column #"ledge_genes"
    #input_df[gene_col] = input_df[gene_col].astype(str)
    input_df[gene_col] = input_df[gene_col].str.split(seperator)
    term_vals = list(input_df[term_col].unique())
    uni_val = list(input_df.index.unique())
    sim_mat = pd.DataFrame(index=uni_val, columns=uni_val)
    exploded_df = input_df.explode(gene_col)
    arr = np.array(input_df[gene_col])
    vals = list(exploded_df[nested_gene_column])
    import scipy.sparse as sparse
    def binarise(sets, full_set):
        """
        General description.

        Parameters:

        Returns: sparse binary matrix of given sets.

        """
        return sparse.csr_matrix([[x in s for x in full_set] for s in sets])
    sparse_matrix = binarise(arr, vals)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sparse_matrix)
    sim_mat = pd.DataFrame(similarities)
    sim_mat.index = uni_val
    sim_mat.columns = uni_val
    return(sim_mat)

#Creating similarity matrix from GO terms
def create_sim_matrix_from_term2(input_df,nested_gene_column="Term",seperator=" ",term_col="Term"):
    """
    General description.

    Parameters:

    Returns:

    """
#    Horrifically bad cosine similairty estimate for word frequency
#    Check term col in columns else, check index as it\s sometimes heree
    if not term_col in list(input_df.columns):
        input_df[term_col] = input_df.index
    input_df = input_df.copy()
    gene_col = nested_gene_column #"ledge_genes"
    #input_df[gene_col] = input_df[gene_col].astype(str)
    term_vals = list(input_df[term_col].unique())
    input_df[gene_col] = input_df[gene_col].str.split(seperator)
    uni_val = list(input_df.index.unique())
    sim_mat = pd.DataFrame(index=uni_val, columns=uni_val)
    exploded_df = input_df.explode(gene_col)

    nan_value = float("NaN")
    exploded_df.replace("", nan_value, inplace=True)
    exploded_df.dropna(subset = [gene_col], inplace=True)
    arr = np.array(input_df[gene_col])

    vals = list(exploded_df[nested_gene_column])

    import scipy.sparse as sparse
    def binarise(sets, full_set):
        """
        General description.

        Parameters:

        Return sparse binary matrix of given sets.

        """
        return sparse.csr_matrix([[x in s for x in full_set] for s in sets])
    sparse_matrix = binarise(arr, vals)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(sparse_matrix)
    sim_mat = pd.DataFrame(similarities)
    sim_mat.index = uni_val
    sim_mat.columns = uni_val
    return(sim_mat)
    #Example usage : sim_mat = create_sim_matrix_from_enr(enr.res2d)
    
def analyze_and_plot_feat_gsea(top_loadings_lowdim, class_name, max_len=20, pre_ranked=True, database='GO_Biological_Process_2021', cutoff=0.25, min_s=5):
    """
    General description.

    Analyzes and plots the top loadings of a low-dimensional model.

    Parameters:
    top_loadings_lowdim (pd.DataFrame): A dataframe containing the top loadings for each class.
    class_name (str): The name of the class to analyze and plot.
    max_len (int): The maximum number of features to plot.
    pre_ranked (bool): Whether the data is pre-ranked.
    database (str): The name of the database to query for enrichment analysis.
    cutoff (float): The cutoff value for enrichment analysis.
    min_s (int): The minimum number of genes in a category to consider for enrichment analysis.

    Returns:
    None
    """
    # Filter the top loadings for the given class
    top_loadings_u = top_loadings_lowdim[top_loadings_lowdim['class'] == class_name].head(max_len)
    top_loadings_u['gene'] = top_loadings_u['feature']

    # Perform enrichment analysis
    if not pre_ranked == True:
        glist = as_gene_list(top_loadings_u, "gene")
        enr = enrich_no_rank(glist, [database])
    else:
        enr = pre_ranked_enr(top_loadings_u, ["gene", "e^coef"], permutation_num=1000, database=database, cutoff=cutoff, min_s=min_s)

    # Print the enrichment score range
    print("Normalised enrichment score ranges, for continuous phenotype tests, a positive value indicates correlation with phenotype profile or top of the ranked list, negative values indicate inverse correlation with profile or correlation with the bottom of the list")
    print(enr.res2d.shape)
    # Plot the enrichment results
    terms = enr.res2d.Term
    axs = enr.plot(terms=terms[1:10], legend_kws={'loc': (1.2, 0)}, show_ranking=True, figsize=(3, 4))
    plt.show()