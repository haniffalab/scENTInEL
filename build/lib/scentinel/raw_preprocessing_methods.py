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

import numpy as np
import seaborn as sns

sns.set_theme(color_codes=True)
# raw_data_feature_selection_dimensionality_reduction


# Feature selection by dispersion
def subset_top_hvgs(adata_lognorm, n_top_genes):
    """
    General description.

    Parameters:

    Returns:

    """
    dispersion_norm = adata_lognorm.var["dispersions_norm"].values.astype("float32")

    dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
    dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower

    disp_cut_off = dispersion_norm[n_top_genes - 1]
    gene_subset = adata_lognorm.var["dispersions_norm"].values >= disp_cut_off
    return adata_lognorm[:, gene_subset]


# Prep data for VAE-based dim reduction
def prep_scVI(adata, n_hvgs=5000, remove_cc_genes=True, remove_tcr_bcr_genes=False):
    """
    General description.

    Parameters:

    Returns:

    """
    ## Remove cell cycle genes
    if remove_cc_genes:
        adata = panfetal_utils.remove_geneset(adata, genes.cc_genes)

    ## Remove TCR/BCR genes
    if remove_tcr_bcr_genes:
        adata = panfetal_utils.remove_geneset(adata, genes.IG_genes)
        adata = panfetal_utils.remove_geneset(adata, genes.TCR_genes)

    ## HVG selection
    adata = subset_top_hvgs(adata, n_top_genes=n_hvgs)
    return adata
