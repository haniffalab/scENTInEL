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
# Utils

# resource usage logger
class DisplayCPU(threading.Thread):
    def run(self):
        """
        General description.

        Parameters:

        Returns:

        """
        tracemalloc.start()
        starting, starting_peak = tracemalloc.get_traced_memory()
        self.running = True
        self.starting = starting
        currentProcess = psutil.Process()
        cpu_pct = []
        peak_cpu = 0
        while self.running:
            peak_cpu = 0
#           time.sleep(3)
#             print('CPU % usage = '+''+ str(currentProcess.cpu_percent(interval=1)))
#             cpu_pct.append(str(currentProcess.cpu_percent(interval=1)))
            cpu = currentProcess.cpu_percent()
        # track the peak utilization of the process
            if cpu > peak_cpu:
                peak_cpu = cpu
                peak_cpu_per_core = peak_cpu/psutil.cpu_count()
        self.peak_cpu = peak_cpu
        self.peak_cpu_per_core = peak_cpu_per_core
        
    def stop(self):
        """
        General description.

        Parameters:

        Returns:

        """
#        cpu_pct = DisplayCPU.run(self)
        self.running = False
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return current, peak

# Frequency redistribution mode for assigning classes by categorical detected communities
def freq_redist_68CI(adata,clusters_reassign):
    """
    General description.

    Parameters:

    Returns:

    """
    if freq_redist != False:
        print('Frequency redistribution commencing')
        cluster_prediction = "consensus_clus_prediction"
        lr_predicted_col = 'predicted'
        pred_out[clusters_reassign] = adata.obs[clusters_reassign].astype(str)
        reassign_classes = list(pred_out[clusters_reassign].unique())
        lm = 1 # lambda value
        pred_out[cluster_prediction] = pred_out[clusters_reassign]
        for z in pred_out[clusters_reassign][pred_out[clusters_reassign].isin(reassign_classes)].unique():
            df = pred_out
            df = df[(df[clusters_reassign].isin([z]))]
            df_count = pd.DataFrame(df[lr_predicted_col].value_counts())
            # Look for classificationds > 68CI
            if len(df_count) > 1:
                df_count_temp = df_count[df_count[lr_predicted_col]>int(int(df_count.mean()) + (df_count.std()*lm))]
                if len(df_count_temp >= 1):
                    df_count = df_count_temp
            #print(df_count)     
            freq_arranged = df_count.index
            cat = freq_arranged[0]
        #Make the cluster assignment first
            pred_out[cluster_prediction] = pred_out[cluster_prediction].astype(str)
            pred_out.loc[pred_out[clusters_reassign] == z, [cluster_prediction]] = cat
        # Create assignments for any classification >68CI
            for cats in freq_arranged:
                #print(cats)
                cats_assignment = cats#.replace(data1,'') + '_clus_prediction'
                pred_out.loc[(pred_out[clusters_reassign] == z) & (pred_out[lr_predicted_col] == cats),[cluster_prediction]] = cats_assignment
        min_counts = pd.DataFrame((pred_out[cluster_prediction].value_counts()))
        reassign = list(min_counts.index[min_counts[cluster_prediction]<=2])
        pred_out[cluster_prediction] = pred_out[cluster_prediction].str.replace(str(''.join(reassign)),str(''.join(pred_out.loc[pred_out[clusters_reassign].isin(list(pred_out.loc[(pred_out[cluster_prediction].isin(reassign)),clusters_reassign])),lr_predicted_col].value_counts().head(1).index.values)))
        return pred_out
    
# Module to produce report for projection accuracy metrics on tranductive runs     
def report_f1(model,train_x, train_label):
    """
    General description.

    Parameters:

    Returns:

    """
    ## Report accuracy score
    
    # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)
    # # evaluate the model and collect the scores
    # n_scores = cross_val_score(lr, train_x, train_label, scoring='accuracy', cv=cv, n_jobs=-1)
    # # report the model performance
    # print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    # Report Precision score
    metric = pd.DataFrame((metrics.classification_report(train_label, model.predict(train_x), digits=2,output_dict=True))).T
    cm = confusion_matrix(train_label, model.predict(train_x))
    #cm = confusion_matrix(train_label, model.predict_proba(train_x))
    df_cm = pd.DataFrame(cm, index = model.classes_,columns = model.classes_)
    df_cm = (df_cm / df_cm.sum(axis=0))*100
    plt.figure(figsize = (20,15))
    sn.set(font_scale=1) # for label size
    pal = sns.diverging_palette(240, 10, n=10)
    #plt.suptitle(('Mean Accuracy 5 fold: %.3f std: %.3f' % (np.mean(n_scores),  np.std(n_scores))), y=1.05, fontsize=18)
    #Plot precision recall and recall
    table = plt.table(cellText=metric.values,colWidths = [1]*len(metric.columns),
    rowLabels=metric.index,
    colLabels=metric.columns,
    cellLoc = 'center', rowLoc = 'center',
    loc='bottom', bbox=[0.25, -0.6, 0.5, 0.3])
    table.scale(1, 2)
    table.set_fontsize(10)

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap=pal) # font size
    print(metrics.classification_report(train_label, model.predict(train_x), digits=2))

def compute_label_log_losses(df, true_label, pred_columns):
    """
    Compute log loss (cross-entropy loss).

    Parameters:
    df : dataframe containing the predicted probabilities and original labels as columns
    true_label : column or array-like of shape (n_samples,) containg cateogrical labels
    pred_columns : columns or array-like of shape (n_samples, n_clases) containg predicted probabilities

    converts to:
    y_true : array-like of shape (n_samples,) True labels. The binary labels in a one-vs-rest fashion.
    y_pred : array-like of shape (n_samples, n_classes) Predicted probabilities. 

    Returns:
    log_loss : dictionary of celltype key and float
    weights : float
    """
    log_losses = {}
    y_true = (pd.get_dummies(df[true_label]))
    y_pred = df[pred_col]
    loss = log_loss(np.array(y_true), np.array(y_pred))
    for label in range(y_true.shape[1]):
        log_loss_label = log_loss(np.array(y_true)[:, label], np.array(y_pred)[:, label])
        log_losses[list(y_true.columns)[label]] = (log_loss_label)
    weights = 1/np.array(list(log_losses.values()))
    weights /= np.sum(weights)
    weights = np.array(weights)
    return loss, log_losses, weights

def regression_results(df, true_label, pred_label, pred_columns):
    """
    General description.

    Parameters:

    Returns:

    """
    # Regression metrics
    y_true = df[true_label]
    y_pred = df[pred_label]
    loss, log_losses, weights = compute_label_log_losses(df, true_label, pred_columns)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
#     r2=metrics.r2_score(y_true, y_pred)
    print('Cross entropy loss: ', round(loss,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    print('label Cross entropy loss: ')
    print(log_losses)  
    return loss, log_losses, weights


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