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
        enr = pre_ranked_enr(top_loadings_u, ["gene", "weighted_impact"], permutation_num=1000, database=database, cutoff=cutoff, min_s=min_s)

    # Print the enrichment score range
    print("Normalised enrichment score ranges, for continuous phenotype tests, a positive value indicates correlation with phenotype profile or top of the ranked list, negative values indicate inverse correlation with profile or correlation with the bottom of the list")
    print(enr.res2d.shape)
    # Plot the enrichment results
    terms = enr.res2d.Term
    axs = enr.plot(terms=terms[1:10], legend_kws={'loc': (1.2, 0)}, show_ranking=True, figsize=(3, 4))
    plt.show()