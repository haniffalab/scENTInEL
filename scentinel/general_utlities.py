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
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
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
    pal = sn.diverging_palette(240, 10, n=10)
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
    y_pred = df[pred_columns]
    
    # Get all unique classes from the true labels
    unique_classes = np.sort(df[true_label].unique())
    # Convert categorical true labels to one-hot encoding
    y_true = pd.get_dummies(df[true_label], columns=unique_classes)
    # Make sure y_pred has columns for all classes in y_true, fill missing with zeros
    y_pred = df[pred_columns].reindex(columns=unique_classes, fill_value=0)
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
    le = LabelEncoder()
    y_true = le.fit_transform(df[true_label])
    y_pred = le.transform(df[pred_label])
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

