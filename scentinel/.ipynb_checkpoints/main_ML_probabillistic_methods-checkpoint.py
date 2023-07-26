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
# main_probabillistic_training_projection_modules
    
# projection module
def reference_projection(adata, model, dyn_std,partial_scale,train_x_partition):
    """
    General description.

    Parameters:

    Returns:

    """
    
    class adata_temp:
        pass
    from sklearn.preprocessing import StandardScaler
    print('Determining model flavour')
    try:
        model_lr =  model['Model']
        print('Consuming celltypist model')
    except:# hasattr(model, 'coef_'):
        print('Consuming non-celltypist model')
        model_lr =  model
    print(model_lr)
    if train_x_partition == 'X':
        print('Matching reference genes in the model')
        k_x = np.isin(list(adata.var.index), list(model_lr.features))
        if k_x.sum() == 0:
            raise ValueError(f"🛑 No features overlap with the model. Please provide gene symbols")
        print(f"🧬 {k_x.sum()} features used for prediction")
        #slicing adata
        k_x_idx = np.where(k_x)[0]
        # adata_temp = adata[:,k_x_idx]
        adata_temp.var = adata[:,k_x_idx].var
        adata_temp.X = adata[:,k_x_idx].X
        adata_temp.obs = adata[:,k_x_idx].obs
        lr_idx = pd.DataFrame(model_lr.features, columns=['features']).reset_index().set_index('features').loc[list(adata_temp.var.index)].values
        # adata_arr = adata_temp.X[:,list(lr_idexes['index'])]
        # slice and reorder model
        ni, fs, cf = model_lr.n_features_in_, model_lr.features, model_lr.coef_
        model_lr.n_features_in_ = lr_idx.size
        model_lr.features = np.array(model_lr.features)[lr_idx]
        model_lr.coef_ = np.squeeze(model_lr.coef_[:,lr_idx]) #model_lr.coef_[:, lr_idx]
        if partial_scale == True:
            print('scaling input data, default option is to use incremental learning and fit in mini bulks!')
            # Partial scaling alg
            scaler = StandardScaler(with_mean=False)
            n = adata_temp.X.shape[0]  # number of rows
            # set dyn scale packet size
            x_len = len(adata_temp.var)
            y_len = len(adata.obs)
            if y_len < 100000:
                dyn_pack = int(x_len/10)
                pack_size = dyn_pack
            else:
                # 10 pack for every 100,000
                dyn_pack = int((y_len/100000)*10)
                pack_size = int(x_len/dyn_pack)
            batch_size =  1000#pack_size#500  # number of rows in each call to partial_fit
            index = 0  # helper-var
            while index < n:
                partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
                partial_x = adata_temp.X[index:index+partial_size]
                scaler.partial_fit(partial_x)
                index += partial_size
            adata_temp.X = scaler.transform(adata_temp.X)
    # model projections
    print('Starting reference projection!')
    if train_x_partition == 'X':
        train_x = adata_temp.X
        pred_out = pd.DataFrame(model_lr.predict(train_x),columns = ['predicted'],index = list(adata.obs.index))
        proba =  pd.DataFrame(model_lr.predict_proba(train_x),columns = model_lr.classes_,index = list(adata.obs.index))
        pred_out = pred_out.join(proba)
        
    elif train_x_partition in list(adata.obsm.keys()): 
        print('{low_dim: this partition modality is still under development!}')
        train_x = adata.obsm[train_x_partition]
        pred_out = pd.DataFrame(model_lr.predict(train_x),columns = ['predicted'],index = list(adata.obs.index))
        proba =  pd.DataFrame(model_lr.predict_proba(train_x),columns = model_lr.classes_,index = list(adata.obs.index))
        pred_out = pred_out.join(proba)
    
    else:
        print('{this partition modality is still under development!}')
    ## insert modules for low dim below

    # Simple dynamic confidence calling
    pred_out['confident_calls'] = pred_out['predicted']
    pred_out.loc[pred_out.max(axis=1)<(pred_out.mean(axis=1) + (1*pred_out.std(axis=1))),'confident_calls'] = pred_out.loc[pred_out.max(axis=1)<(pred_out.mean(axis=1) + (1*pred_out.std(axis=1))),'confident_calls'].astype(str) + '_uncertain'
    # means_ = self.model.scaler.mean_[lr_idx] if self.model.scaler.with_mean else 0
    return(pred_out,train_x,model_lr,adata_temp)

# Modified LR train module, does not work with low-dim by default anymore, please use low-dim adapter
def LR_train(adata, train_x, train_label, penalty='elasticnet', sparcity=0.2,max_iter=200,l1_ratio =0.2,tune_hyper_params =False,n_splits=5, n_repeats=3,l1_grid = [0.01,0.2,0.5,0.8], c_grid = [0.01,0.2,0.4,0.6]):
    """
    General description.

    Parameters:

    Returns:

    """
    if tune_hyper_params == True:
        train_labels=train_label
        results = tune_lr_model(adata, train_x_partition = train_x, random_state = 42,  train_labels = train_labels, n_splits=n_splits, n_repeats=n_repeats,l1_grid = l1_grid, c_grid = c_grid)
        print('hyper_params tuned')
        sparcity = results.best_params_['C']
        l1_ratio = results.best_params_['l1_ratio']
    
    lr = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  max_iter, n_jobs=thread_num)
    if (penalty == "l1"):
        lr = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  max_iter, dual = True, solver = 'liblinear',multi_class = 'ovr', n_jobs=thread_num ) # one-vs-rest
    if (penalty == "elasticnet"):
        lr = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  max_iter, dual=False,solver = 'saga',l1_ratio=l1_ratio,multi_class = 'ovr', n_jobs=thread_num)
    if train_x == 'X':
        subset_train = adata.obs.index
        # Define training parameters
        train_label = adata.obs[train_label].values
#        predict_label = train_label[subset_predict]
#        train_label = train_label[subset_train]
        train_x = adata.X#[adata.obs.index.isin(list(adata.obs[subset_train].index))]
#        predict_x = adata.X[adata.obs.index.isin(list(adata.obs[subset_predict].index))]
    elif train_x in adata.obsm.keys():
        # Define training parameters
        train_label = adata.obs[train_label].values
#        predict_label = train_label[subset_predict]
#         train_label = train_label[subset_train]
        train_x = adata.obsm[train_x]
#        predict_x = train_x
#        train_x = train_x[subset_train, :]
        # Define prediction parameters
#        predict_x = predict_x[subset_predict]
#        predict_x = pd.DataFrame(predict_x)
#        predict_x.index = adata.obs[subset_predict].index
    # Train predictive model using user defined partition labels (train_x ,train_label, predict_x)
    model = lr.fit(train_x, train_label)
    model.features = np.array(adata.var.index)
    return model

def tune_lr_model(adata, train_x_partition = 'X', random_state = 42, use_bayes_opt=True, train_labels = None, n_splits=5, n_repeats=3,l1_grid = [0.1,0.2,0.5,0.8], c_grid = [0.1,0.2,0.4,0.6]):
    """
    General description.

    Parameters:

    Returns:

    """
    import bless as bless
    from sklearn.gaussian_process.kernels import RBF
    from numpy import arange
    from sklearn.model_selection import RepeatedKFold
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GridSearchCV
    from skopt import BayesSearchCV

    # If latent rep is provided, randomly sample data in spatially aware manner for initialisation
    r = np.random.RandomState(random_state)
    if train_x_partition in adata.obsm.keys():
        lvg = bless.bless(tune_train_x, RBF(length_scale=20), lam_final = 2, qbar = 2, random_state = r, H = 10, force_cpu=True)
    #     try:
    #         import cupy
    #         lvg_2 = bless(adata.obsm[train_x_partition], RBF(length_scale=10), 10, 10, r, 10, force_cpu=False)
    #     except ImportError:
    #         print("cupy not found, defaulting to numpy")
        adata_tuning = adata[lvg.idx]
        tune_train_x = adata_tuning.obsm[train_x_partition][:]
    else:
        print('no latent representation provided, random sampling instead')
        prop = 0.1
        random_vertices = []
        n_ixs = int(len(adata.obs) * prop)
        random_vertices = random.sample(list(range(len(adata.obs))), k=n_ixs)
        adata_tuning = adata[random_vertices]
        tune_train_x = adata_tuning.X
        
    if not train_labels == None:
        tune_train_label = adata_tuning.obs[train_labels]
    elif train_labels == None:
        try:
            print('no training labels provided, defaulting to unsuperived leiden clustering, updates will change this to voronoi greedy sampling')
            sc.tl.leiden(adata_tuning)
        except:
            print('no training labels provided, no neighbors, defaulting to unsuperived leiden clustering, updates will change this to voronoi greedy sampling')
            sc.pp.neighbors(adata_hm, n_neighbors=15, n_pcs=50)
            sc.tl.leiden(adata_tuning)
        tune_train_label = adata_tuning.obs['leiden']
    ## tune regularization for multinomial logistic regression
    print('starting tuning loops')
    X = tune_train_x
    y = tune_train_label
    grid = dict()
    # define model
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    #model = LogisticRegression(penalty = penalty, max_iter =  200, dual=False,solver = 'saga', multi_class = 'multinomial',)
    model = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  100, n_jobs=4)
    if (penalty == "l1"):
        model = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  100, dual = True, solver = 'liblinear',multi_class = 'multinomial', n_jobs=4 ) # one-vs-rest
    if (penalty == "elasticnet"):
        model = LogisticRegression(penalty = penalty, C = sparcity, max_iter =  100, dual=False,solver = 'saga',l1_ratio=l1_ratio,multi_class = 'multinomial', n_jobs=4) # use multinomial class if probabilities are descrete
        grid['l1_ratio'] = l1_grid
    grid['C'] = c_grid
    
    if use_bayes_opt == True:
        # define search space
        search_space = {'C': (np.min(c_grid), np.max(c_grid), 'log-uniform'), 
                        'l1_ratio': (np.min(l1_grid), np.max(l1_grid), 'uniform') if 'elasticnet' in penalty else None}
        # define search
        search = BayesSearchCV(model, search_space, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # perform the search
        results = search.fit(X, y)
    else:
        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # perform the search
        results = search.fit(X, y)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    return results

def prep_training_data(adata_temp,feat_use,batch_key, model_key, batch_correction=False, var_length = 7500,penalty='elasticnet',sparcity=0.2,max_iter = 200,l1_ratio = 0.1,partial_scale=True,train_x_partition ='X',theta = 3,tune_hyper_params=False ):
    """
    General description.

    Parameters:

    Returns:

    """
    model_name = model_key + '_lr_model'
    print('performing highly variable gene selection')
    sc.pp.highly_variable_genes(adata_temp, batch_key = batch_key, subset=False)
    adata_temp = subset_top_hvgs(adata_temp,var_length)
    #scale the input data
    if partial_scale == True:
        print('scaling input data, default option is to use incremental learning and fit in mini bulks!')
        # Partial scaling alg
        #adata_temp.X = (adata_temp.X)
        scaler = StandardScaler(with_mean=False)
        n = adata_temp.X.shape[0]  # number of rows
        # set dyn scale packet size
        x_len = len(adata_temp.var)
        y_len = len(adata_temp.obs)
        if y_len < 100000:
            dyn_pack = int(x_len/10)
            pack_size = dyn_pack
        else:
            # 10 pack for every 100,000
            dyn_pack = int((y_len/100000)*10)
            pack_size = int(x_len/dyn_pack)
        batch_size =  1000#pack_size#500  # number of rows in each call to partial_fit
        index = 0  # helper-var
        while index < n:
            partial_size = min(batch_size, n - index)  # needed because last loop is possibly incomplete
            partial_x = adata_temp.X[index:index+partial_size]
            scaler.partial_fit(partial_x)
            index += partial_size
        adata_temp.X = scaler.transform(adata_temp.X)
#     else:
#         sc.pp.scale(adata_temp, zero_center=True, max_value=None, copy=False, layer=None, obsm=None)
    if (train_x_partition != 'X') & (train_x_partition in adata_temp.obsm.keys()):
        print('train partition is not in OBSM, defaulting to PCA')
        # Now compute PCA
        sc.pp.pca(adata_temp, n_comps=100, use_highly_variable=True, svd_solver='arpack')
        sc.pl.pca_variance_ratio(adata_temp, log=True,n_pcs=100)
        
        # Batch correction options
        # The script will test later which Harmony values we should use 
        if(batch_correction == "Harmony"):
            print("Commencing harmony")
            adata_temp.obs['lr_batch'] = adata_temp.obs[batch_key]
            batch_var = "lr_batch"
            # Create hm subset
            adata_hm = adata_temp[:]
            # Set harmony variables
            data_mat = np.array(adata_hm.obsm["X_pca"])
            meta_data = adata_hm.obs
            vars_use = [batch_var]
            # Run Harmony
            ho = hm.run_harmony(data_mat, meta_data, vars_use,theta=theta)
            res = (pd.DataFrame(ho.Z_corr)).T
            res.columns = ['X{}'.format(i + 1) for i in range(res.shape[1])]
            # Insert coordinates back into object
            adata_hm.obsm["X_pca_back"]= adata_hm.obsm["X_pca"][:]
            adata_hm.obsm["X_pca"] = np.array(res)
            # Run neighbours
            #sc.pp.neighbors(adata_hm, n_neighbors=15, n_pcs=50)
            adata_temp = adata_hm[:]
            del adata_hm
        elif(batch_correction == "BBKNN"):
            print("Commencing BBKNN")
            sc.external.pp.bbknn(adata_temp, batch_key=batch_var, approx=True, metric='angular', copy=False, n_pcs=50, trim=None, n_trees=10, use_faiss=True, set_op_mix_ratio=1.0, local_connectivity=15) 
        print("adata1 and adata2 are now combined and preprocessed in 'adata' obj - success!")


    # train model
#    train_x = adata_temp.X
    #train_label = adata_temp.obs[feat_use]
    print('proceeding to train model')
    model = LR_train(adata_temp, train_x = train_x_partition, train_label=feat_use, penalty=penalty, sparcity=sparcity,max_iter=max_iter,l1_ratio = l1_ratio,tune_hyper_params = tune_hyper_params)
    model.features = list(adata_temp.var.index)
    return model

def compute_weighted_impact(varm_file, top_loadings, threshold=0.05):
    """
    General description.
    Computes the weighted impact of the features of a low-dimensional model.

    Parameters:
    varm_file (str): The path to the file containing the variable loadings of the model.
    top_loadings (pd.DataFrame): A dataframe containing the top loadings for each class.
    threshold (float): The p-value threshold for significance.

    Returns:
    top_loadings_lowdim (pd.DataFrame): A dataframe containing the top weighted impacts for each class.
    """
    # Load the variable loadings from the file
    model_varm = pd.read_csv(varm_file, index_col=0)

    # Map the feature names to the column names of the variable loadings
    feature_set = dict(zip(sorted(top_loadings['feature'].unique()), model_varm.columns))

    # Melt the variable loadings dataframe and add a column for p-values
    varm_melt = pd.melt(model_varm.reset_index(), id_vars='index')
    varm_melt['pvals'] = np.nan

    # Compute the p-values for each variable
    for variable in varm_melt['variable'].unique():
        varm_loadings = varm_melt[varm_melt['variable'] == variable]
        med = np.median(varm_loadings['value'])
        mad = np.median(np.abs(varm_loadings['value'] - med))
        pvals = scipy.stats.norm.sf(varm_loadings['value'], loc=med, scale=1.4826*mad)
        varm_melt.loc[varm_melt['variable'] == variable, 'pvals'] = pvals

    # Filter the variables based on the p-value threshold
    varm_sig = varm_melt[varm_melt['pvals'] < threshold]

    # Compute the weighted impact for each feature of each class
    top_loadings_lowdim = pd.DataFrame(columns=['class', 'feature', 'weighted_impact', 'e^coef_pval', 'e^coef', 'is_significant_sf'])
    top_loadings_lw = top_loadings.groupby('class').head(10)
    top_loadings_lw['feature'] = top_loadings_lw['feature'].map(feature_set)

    for classes in top_loadings_lw['class'].unique():
        for feature in top_loadings_lw.loc[top_loadings_lw['class'] == classes, ['feature', 'e^coef']].values:
            temp_varm_sig = varm_sig[varm_sig['variable'] == feature[0]]
            temp_varm_sig['weighted_impact'] = temp_varm_sig['value'] * feature[1]
            temp_varm_sig = temp_varm_sig[['index', 'weighted_impact']]
            temp_varm_sig.columns = ['feature', 'weighted_impact']
            temp_varm_sig['class'] = classes
            temp_varm_sig['e^coef_pval'] = top_loadings_lw.loc[(top_loadings_lw['class'] == classes) & (top_loadings_lw['feature'] == feature[0]), 'e^coef_pval'].values[0]
            temp_varm_sig['e^coef'] = top_loadings_lw.loc[(top_loadings_lw['class'] == classes) & (top_loadings_lw['feature'] == feature[0]), 'e^coef'].values[0]
            temp_varm_sig['is_significant_sf'] = top_loadings_lw.loc[(top_loadings_lw['class'] == classes) & (top_loadings_lw['feature'] == feature[0]), 'is_significant_sf'].values[0]
            top_loadings_lowdim = pd.concat([top_loadings_lowdim, temp_varm_sig], ignore_index=True)
    # Return the top 100 features with the highest weighted impact for each class
    top_loadings_lowdim = top_loadings_lowdim.sort_values('weighted_impact', ascending=False).groupby('class').head(100)
    return top_loadings_lowdim


def get_binary_neigh_matrix(connectivities):
    """
    General description:
    Converts the connectivities matrix to a binary neighborhood membership matrix.
    """
    return (connectivities > 0).astype(int)

def get_label_counts(neigh_matrix, labels):
    """
    General description.
    Counts the number of occurrences of each label in the neighborhood of each cell.
    """
    return pd.DataFrame(neigh_matrix.T.dot(pd.get_dummies(labels)))

def compute_dist_entropy_product(neigh_membership, labels, dist_matrix):
    """
    General description.
    Computes the product of the average neighborhood distance and the entropy
    of the label distribution in the neighborhood for each cell and each label.
    """
    # Count the occurrences of each label in the neighborhood of each cell
    label_counts = get_label_counts(neigh_membership, labels)

    # Compute the entropy of the label distribution in the neighborhood of each cell
    entropy_values = label_counts.apply(entropy, axis=1)

    # Compute the average neighborhood distance for each cell
    avg_distances = dist_matrix.multiply(neigh_membership).mean(axis=1).A1

    # Compute the product of the average distance and the entropy for each cell
    dist_entropy_product = avg_distances * entropy_values

    return dist_entropy_product

class WeightsOutput:
    def __init__(self, weights, rhats, means, sds):
        """
        General description.

        Parameters:

        Returns:

        """
        self.weights = weights
        self.rhats = rhats
        self.means = means
        self.sds = sds

def compute_weights(adata, use_rep, original_labels_col, predicted_labels_col):
    """
    General description.

    Parameters:

    Returns:

    """
    # Extract the necessary data from the anndata object
    obs_met = adata.obs
    neigh_membership = get_binary_neigh_matrix(adata.obsp[adata.uns[use_rep]['connectivities_key']])
    original_labels = obs_met[original_labels_col]
    predicted_labels = obs_met[predicted_labels_col]
    dist_matrix = adata.obsp[adata.uns[use_rep]['distances_key']]

    # Compute the 'distance-entropy' product for each cell and each label
    dist_entropy_product = compute_dist_entropy_product(neigh_membership, predicted_labels, dist_matrix)

    # Compute the 'distance-entropy' product for the original labels
    dist_entropy_product_orig = compute_dist_entropy_product(neigh_membership, original_labels, dist_matrix)

    weights = {}
    rhat_values = {}
    means = []  # Collect all posterior means
    sds = []  # Collect all posterior standard deviations
    for label in np.unique(predicted_labels):
        print("Sampling {} posterior distribution".format(label))
        # Perform Bayesian inference to compute the posterior distribution of the
        # 'distance-entropy' product for this label
        orig_pos = obs_met[original_labels_col].isin([label])
        pred_pos = obs_met[predicted_labels_col].isin([label])
        with pm.Model() as model:
            #priors
            mu = pm.Normal('mu', mu=dist_entropy_product_orig[orig_pos.values].mean(), sd=dist_entropy_product_orig[orig_pos.values].std())
            sd = pm.HalfNormal('sd', sd=dist_entropy_product_orig[orig_pos.values].std())
            #observations
            obs = pm.Normal('obs', mu=mu, sd=sd, observed=dist_entropy_product_orig[pred_pos.values])
            
            if len(orig_pos) > 10000:
                samp_rate = 0.1
                smp = int(len(orig_pos)*samp_rate)
                tne = smp = int(len(orig_pos)*samp_rate)/2
                trace = pm.sample(smp, tune=tne)
            else:
                trace = pm.sample(1000, tune=500)
        # Compute R-hat for this label
        rhat = pm.rhat(trace)
        rhat_values[label] = {var: rhat[var].data for var in rhat.variables}
        # Compute the mean and the standard deviation of the posterior distribution for this label
        mean_posterior = pm.summary(trace)['mean']['mu']
        sd_posterior = pm.summary(trace)['sd']['sd']
        sds.append(sd_posterior)
        means.append(mean_posterior)
        
    # Mean posterior probabilitty models the stability of a label given entropy_distance measures within it's neighborhood
    max_mean = max(means)
    # SD here models the uncertainty of label entropy_distance measures
    max_sd = max(sds)  # Compute the maximum standard deviation
    
    # Compute the weights as the sum of the normalized mean and the normalized standard deviation. This makes each weight relative to each other
    # shift all weights up by epiislon constant
    epsilon = 0.01
    for label, mean, sd in zip(np.unique(predicted_labels), means, sds):
        weights[label] = (1 - mean / max_mean) * (1 - sd / max_sd) + epsilon

    return WeightsOutput(weights, rhat_values, means, sds)

def apply_weights(prob_df, weights):
    """
    General description.
    Applies the computed weights to the probability dataframe and normalizes the result.
    Parameters:
    prob_df (pd.DataFrame): A dataframe where each row corresponds to a cell and each column corresponds to a label. Each entry is the probability of the cell being of the label.
    weights (dict): A dictionary where each key-value pair corresponds to a label and its weight.

    Returns:
    norm_df (pd.DataFrame): A dataframe of the same shape as prob_df, but with the probabilities weighted and normalized.
    """
    # Apply the weights
    weighted_df = prob_df.mul(weights.weights)
    # Normalize the result
    norm_df = weighted_df.div(weighted_df.sum(axis=1), axis=0)
    return norm_df