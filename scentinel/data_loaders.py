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
# libraries

import pickle as pkl
from pathlib import Path

import requests
import scanpy as sc
import seaborn as sns

sns.set_theme(color_codes=True)
# Data_model_loaders


def load_models(model_dict, model_run):
    """
    General description.

    Parameters:

    Returns:

    """
    if (Path(model_dict[model_run])).is_file():
        # Load data (deserialize)
        model = pkl.load(open(model_dict[model_run], "rb"))
        return model
    elif "http" in model_dict[model_run]:
        print("Loading model from web source")
        r_get = requests.get(model_dict[model_run])
        fpath = "./model_temp.sav"
        open(fpath, "wb").write(r_get.content)
        model = pkl.load(open(fpath, "rb"))
        return model


def load_adatas(
    adatas_dict, data_merge, data_key_use, QC_normalise, backed=False, **kwargs
):
    """
    General description.

    Parameters:

    Returns:

    """
    # unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

    if data_merge == True:
        # Read
        gene_intersect = {}  # unused here
        adatas = {}
        for dataset in adatas_dict.keys():
            if "https" in adatas_dict[dataset]:
                print("Loading anndata from web source")
                adatas[dataset] = sc.read(
                    "./temp_adata.h5ad", backup_url=adatas_dict[dataset]
                )
            adatas[dataset] = sc.read(adatas_dict[dataset], backed=backed)
            adatas[dataset].var_names_make_unique()
            adatas[dataset].obs["dataset_merge"] = dataset
            adatas[dataset].obs["dataset_merge"] = dataset
            gene_intersect[dataset] = list(adatas[dataset].var.index)
        adata = list(adatas.values())[0].concatenate(
            list(adatas.values())[1:], join="inner"
        )
        return adatas, adata
    if not data_merge:
        if "https" in adatas_dict[data_key_use]:
            print("Loading anndata from web source")
            adata = sc.read("./temp_adata.h5ad", backup_url=adatas_dict[data_key_use])
        else:
            adata = sc.read(adatas_dict[data_key_use], backed=backed)
    if QC_normalise == True:
        print(
            "option to apply standardisation to data detected, performing basic QC filtering"
        )
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)

    return adata
