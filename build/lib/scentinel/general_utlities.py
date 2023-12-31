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
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# import pymc3 as pm
from scipy.sparse import csr_matrix
from scipy.stats import entropy
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
# Utils

import h5py
from tqdm import tqdm
from scipy.sparse import vstack
import gc

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
def freq_redist_68CI(pred_out,clusters_reassign):
    """
    General description.

    Parameters:

    Returns:

    """
    freq_redist = clusters_reassign
    if freq_redist != False:
        print('Frequency redistribution commencing')
        cluster_prediction = "consensus_clus_prediction"
        lr_predicted_col = 'predicted'
#         pred_out[clusters_reassign] = adata.obs[clusters_reassign].astype(str)
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
    ## Report accuracy score
    # ...
    # Report Precision score
    predicted_labels = model.predict(train_x)
    unique_labels = np.unique(np.concatenate((train_label, predicted_labels)))
    metric = pd.DataFrame(classification_report(train_label, predicted_labels, digits=2,output_dict=True)).T
    cm = confusion_matrix(train_label, predicted_labels, labels=unique_labels)
    df_cm = pd.DataFrame(cm, index = unique_labels, columns = unique_labels)
    df_cm = (df_cm / df_cm.sum(axis=0))*100
    plt.figure(figsize = (20,15))
    sns.set(font_scale=1) # for label size
    pal = sns.diverging_palette(240, 10, n=10)
    #Plot precision recall and recall
    num_rows = len(metric.index)
    scale_factor = num_rows * 0.1  # scale factor depends on the number of rows
    bbox_y = -0.4 - num_rows * 0.05  # vertical position of the bbox depends on the number of rows
    bbox_height = num_rows * 0.05  # height of the bbox depends on the number of rows

    table = plt.table(cellText=metric.values, colWidths=[1]*len(metric.columns),
                      rowLabels=metric.index,
                      colLabels=metric.columns,
                      cellLoc='center', rowLoc='center',
                      loc='bottom', bbox=[0.25, bbox_y, 0.5, bbox_height])
    table.scale(1, scale_factor)  # scale the table
    table.set_fontsize(10)


    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap=pal) # font size
    print(classification_report(train_label, predicted_labels, digits=2))
    
    
# Generate psuedocells
def compute_local_scaling_factors(data, neighborhoods_matrix):
    """Compute local scaling factors based on total counts (UMIs) for each neighborhood."""
    total_counts_per_cell = data.sum(axis=1)
    avg_counts_neighborhood = neighborhoods_matrix.dot(total_counts_per_cell) / neighborhoods_matrix.sum(axis=1)
    local_factors = total_counts_per_cell / neighborhoods_matrix.T.dot(avg_counts_neighborhood)
    return local_factors.A1

def compute_global_scaling_factors(data):
    """Compute global scaling factors based on total counts (UMIs) for all cells."""
    avg_counts = data.sum(axis=1).mean()
    return (data.sum(axis=1) / avg_counts).A1

def aggregate_data_single_load(adata, adata_samp, connectivity_matrix, method='local'):
    indices = adata.obs.index.isin(adata_samp.obs.index).nonzero()[0]
    neighborhoods_matrix = connectivity_matrix[indices]
    try:
        adata.to_memory()
    except:
        adata = adata
        
    if not isinstance(adata.X, csr_matrix):
        expression_matrix = csr_matrix(adata.X.toarray())
    else:
        expression_matrix = adata.X

    # Store original counts in dataframe
    orig_obs_counts = pd.DataFrame(index = adata.obs_names, columns=['n_counts'])
    orig_obs_counts['n_counts'] = expression_matrix_chunk.sum(axis=1).A1
    
    # Apply scaling factors to individual cell expression profiles
    if method == 'local':
        factors = compute_local_scaling_factors(expression_matrix, neighborhoods_matrix)
    elif method == 'global':
        factors = compute_global_scaling_factors(expression_matrix)
    else:
        factors = np.ones(expression_matrix.shape[0])

    normalized_data = expression_matrix.multiply(np.reciprocal(factors)[:, np.newaxis])

    # Aggregate the normalized data
    aggregated_data = neighborhoods_matrix.dot(normalized_data)
    
    obs = adata.obs.iloc[indices]
    pseudobulk_adata = sc.AnnData(aggregated_data, obs=obs, var=adata.var)
    
    # Store original data neighbourhood identity
    pseudobulk_adata.uns['orig_data_connectivity_information'] = anndata.AnnData(
        X = adata.obsp["connectivities"],
        obs = pd.DataFrame(index = adata.obs_names),
        var = pd.DataFrame(index = adata.obs_names),
    )
    # Store original counts per cell
    pseudobulk_adata.obs['orig_counts_per_cell'] = orig_obs_counts
    
    # Store connectivity binary assignment 
    pseudobulk_adata.uns['orig_data_connectivity_information'].uns['neighbourhood_identity'] = ((adata.obsp["connectivities"][[adata.obs_names.get_loc(x) for x in pseudo_bulk_data.obs_names], :]) > 0).astype(int)
    
    return pseudobulk_adata

def aggregate_data_v0_1_0(adata, adata_samp, connectivity_matrix, method='local', chunk_size=100):
    """
    Aggregate data in chunks for improved memory efficiency.
    
    Parameters:
    - adata: The main AnnData object containing expression data
    - adata_samp: Subset of AnnData for which pseudocells are created
    - connectivity_matrix: Matrix indicating cell connectivity (e.g., from kNN graph)
    - method: Method for scaling ('local', 'global', or 'none')
    - chunk_size: Number of samples to process in each chunk
    
    Returns:
    - AnnData object with aggregated data
    """

    # Check if in backed mode
    is_backed = adata.isbacked
    if not is_backed and len(adata)<1000000:
        # Use the regular approach if not in backed mode
        print("Data is small enough to proceed with direct dot products")
        return aggregate_data_single_load(adata, adata_samp, connectivity_matrix, method)
    if adata_samp.isbacked:
        adata_samp = adata_samp.to_memory()
    
    print("Data is too large to process in a single view, processing in chunks ")
    # Determine the number of chunks to process
    n_samples = adata_samp.shape[0]
    n_chunks = (n_samples + chunk_size - 1) // chunk_size  # Ceiling division
    aggregated_data_dict = {}
    obs_dict = {}
    
    orig_obs_counts = pd.DataFrame(index = adata.obs_names, columns=['n_counts'])
    
    # Loop through chunks with a progress bar
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        current_chunk = adata_samp[start_idx:end_idx]
        obs_dict[chunk_idx] = adata_samp.obs.iloc[start_idx:end_idx]

        # Get indices of cells in the current chunk
        #indices = adata.obs.index.isin(current_chunk.obs.index).nonzero()[0]
        indices = adata.obs.index.get_indexer(current_chunk.obs.index)
        # Extract the corresponding neighborhood matrix
        neighborhoods_matrix_chunk = connectivity_matrix[indices]
        # Identify unique neighbor indices for cells in this chunk
        neighbor_indices = np.unique(neighborhoods_matrix_chunk.nonzero()[1])

        # Adjust neighborhood matrix to only cover relevant neighbors
        neighborhoods_matrix_chunk = connectivity_matrix[indices, :][:, neighbor_indices]

        # Extract the expression matrix for these neighbors
        expression_matrix_chunk = adata[neighbor_indices].to_memory().X
        
        # Store original counts in dataframe
        orig_obs_counts.loc[adata[neighbor_indices].obs.index, 'n_counts'] = expression_matrix_chunk.sum(axis=1).A1
        
        # Calculate scaling factors based on the specified method
        if method == 'local':
            factors = compute_local_scaling_factors(expression_matrix_chunk, neighborhoods_matrix_chunk)
        elif method == 'global':
            factors = compute_global_scaling_factors(expression_matrix_chunk)
        elif method == 'sum':
            aggregated_data_chunk = eighborhoods_matrix_chunk.dot(expression_matrix_chunk)
        else:
            factors = np.ones(expression_matrix_chunk.shape[0])
            
        if method != 'sum':
            # Normalize data using scaling factors
            normalized_data_chunk = expression_matrix_chunk.multiply(np.reciprocal(factors)[:, np.newaxis])

            # Aggregate the normalized data using the neighborhood matrix
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(normalized_data_chunk)

        # Store in dictionary with chunk_idx as the key
        aggregated_data_dict[chunk_idx] = aggregated_data_chunk
            

        # Delete variables that are not needed to free up memory
        del current_chunk
        del neighborhoods_matrix_chunk
        del expression_matrix_chunk
        del aggregated_data_chunk
        # Suggest to the garbage collector to cleanup
        gc.collect()

    # Combine results from all chunks using ordered indices
    ordered_chunks = sorted(aggregated_data_dict.keys())
    aggregated_data_combined = scipy.sparse.vstack([aggregated_data_dict[idx] for idx in ordered_chunks])
    aggregated_obs = pd.concat([obs_dict[idx] for idx in ordered_chunks], axis=0)
    # Return as AnnData object
    return sc.AnnData(aggregated_data_combined, obs=aggregated_obs, var=adata.var)

def aggregate_data(adata, adata_samp, connectivity_matrix, method='local', chunk_size=100):
    """
    Aggregate data in chunks for improved memory efficiency.
    
    Parameters:
    - adata: The main AnnData object containing expression data
    - adata_samp: Subset of AnnData for which pseudocells are created
    - connectivity_matrix: Matrix indicating cell connectivity (e.g., from kNN graph)
    - method: Method for scaling ('local', 'global', or 'none')
    - chunk_size: Number of samples to process in each chunk
    
    Returns:
    - AnnData object with aggregated data
    """

    # Check if in backed mode
    is_backed = adata.isbacked
    if not is_backed and len(adata)<1000000:
        # Use the regular approach if not in backed mode
        print("Data is small enough to proceed with direct dot products")
        return aggregate_data_single_load(adata, adata_samp, connectivity_matrix, method)
    if adata_samp.isbacked:
        adata_samp = adata_samp.to_memory()
    
    print("Data is too large to process in a single view, processing in chunks ")
    # Determine the number of chunks to process
    n_samples = adata_samp.shape[0]
    n_chunks = (n_samples + chunk_size - 1) // chunk_size  # Ceiling division
    aggregated_data_dict = {}
    obs_dict = {}

    # Loop through chunks with a progress bar
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        current_chunk = adata_samp[start_idx:end_idx]
        obs_dict[chunk_idx] = adata_samp.obs.iloc[start_idx:end_idx]

        # Get indices of cells in the current chunk
        #indices = adata.obs.index.isin(current_chunk.obs.index).nonzero()[0]
        indices = adata.obs.index.get_indexer(current_chunk.obs.index)
        # Extract the corresponding neighborhood matrix
        neighborhoods_matrix_chunk = connectivity_matrix[indices]
        # Identify unique neighbor indices for cells in this chunk
        neighbor_indices = np.unique(neighborhoods_matrix_chunk.nonzero()[1])

        # Adjust neighborhood matrix to only cover relevant neighbors
        neighborhoods_matrix_chunk = connectivity_matrix[indices, :][:, neighbor_indices]

        # Extract the expression matrix for these neighbors
        expression_matrix_chunk = adata[neighbor_indices].to_memory().X

        # Calculate scaling factors based on the specified method
        if method == 'local':
            factors = compute_local_scaling_factors(expression_matrix_chunk, neighborhoods_matrix_chunk)
        elif method == 'global':
            factors = compute_global_scaling_factors(expression_matrix_chunk)
        elif method == 'sum':
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(expression_matrix_chunk)
        else:
            factors = np.ones(expression_matrix_chunk.shape[0])
            
        if method != 'sum':
            # Normalize data using scaling factors
            normalized_data_chunk = expression_matrix_chunk.multiply(np.reciprocal(factors)[:, np.newaxis])

            # Aggregate the normalized data using the neighborhood matrix
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(normalized_data_chunk)

        # Store in dictionary with chunk_idx as the key
        aggregated_data_dict[chunk_idx] = aggregated_data_chunk
            

        # Delete variables that are not needed to free up memory
        del current_chunk
        del neighborhoods_matrix_chunk
        del expression_matrix_chunk
        del aggregated_data_chunk
        # Suggest to the garbage collector to cleanup
        gc.collect()

    # Combine results from all chunks using ordered indices
    ordered_chunks = sorted(aggregated_data_dict.keys())
    aggregated_data_combined = scipy.sparse.vstack([aggregated_data_dict[idx] for idx in ordered_chunks])
    aggregated_obs = pd.concat([obs_dict[idx] for idx in ordered_chunks], axis=0)
    
    # Create aggregated AnnData object
    pseudobulk_adata = sc.AnnData(aggregated_data_combined, obs=aggregated_obs, var=adata.var)
    
    # Store original data neighbourhood identity
    pseudobulk_adata.uns['orig_data_connectivity_information'] = anndata.AnnData(
        X = adata.obsp["connectivities"],
        obs = pd.DataFrame(index = adata.obs_names),
        var = pd.DataFrame(index = adata.obs_names),
    )
    # Store original counts per cell
    pseudobulk_adata.obs['orig_counts_per_cell'] = orig_obs_counts
    
    # Store connectivity binary assignment 
    pseudobulk_adata.uns['orig_data_connectivity_information'].uns['neighbourhood_identity'] = ((adata.obsp["connectivities"][[adata.obs_names.get_loc(x) for x in pseudo_bulk_data.obs_names], :]) > 0).astype(int)

    return pseudobulk_adata