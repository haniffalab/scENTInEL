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

import itertools
import math
import warnings

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sparse
import seaborn as sns

sns.set_theme(color_codes=True)


def long_format_features(top_loadings):
    """
    General description.

    Parameters:

    Returns:
    """
    p = top_loadings.loc[:, top_loadings.columns.str.endswith("_e^coef")]
    p = pd.melt(p)
    n = top_loadings.loc[:, top_loadings.columns.str.endswith("_feature")]
    n = pd.melt(n)
    l = top_loadings.loc[:, top_loadings.columns.str.endswith("_coef")]
    l = pd.melt(l)
    n = n.replace(regex=r"_feature", value="")
    n = n.rename(columns={"variable": "class", "value": "feature"})
    p = (p.drop(["variable"], axis=1)).rename(columns={"value": "e^coef"})
    l = (l.drop(["variable"], axis=1)).rename(columns={"value": "coef"})
    concat = pd.concat([n, p, l], axis=1)
    return concat

    ### Feature importance notes
    # - If we increase the x feature one unit, then the prediction will change e to the power of its weight. We can apply this rule to the all weights to find the feature importance.
    # - We will calculate the Euler number to the power of its coefficient to find the importance.
    # - To sum up an increase of x feature by one unit increases the odds of being versicolor class by a factor of x[importance] when all other features remain the same.
    # - For low-dim, we look at the distribution of e^coef per class, we extract the
    # class coef_extract:
    #     def __init__(self, model,features, pos):


# #         self.w = list(itertools.chain(*(model.coef_[pos]).tolist())) #model.coef_[pos]
#         self.w = model.coef_[class_pred_pos]
#         self.features = features


def model_feature_sf(long_format_feature_importance, coef_use):
    """
    General description.

    Parameters:

    Returns:

    """
    long_format_feature_importance[str(coef_use) + "_pval"] = "NaN"
    for class_lw in long_format_feature_importance["class"].unique():
        df_loadings = long_format_feature_importance[
            long_format_feature_importance["class"].isin([class_lw])
        ]
        comps = coef_use  #'e^coef'
        med = np.median(df_loadings[comps])
        mad = np.median(np.absolute(df_loadings[comps] - np.median(df_loadings[comps])))
        # Survival function scaled by 1.4826 of MAD (approx norm)
        pvals = scipy.stats.norm.sf(
            df_loadings[comps], loc=med, scale=1.4826 * mad
        )  # 95% CI of MAD <10,000 samples
        # pvals = scipy.stats.norm.sf(df_loadings[comps], loc=U, scale=1*std)
        # df_loadings[str(comps) + '_pval'] = pvals
        df_loadings.loc[:, str(comps) + "_pval"] = pvals
        long_format_feature_importance.loc[
            long_format_feature_importance.index.isin(df_loadings.index)
        ] = df_loadings
    long_format_feature_importance["is_significant_sf"] = False
    long_format_feature_importance.loc[
        long_format_feature_importance[coef_use + "_pval"] < 0.05, "is_significant_sf"
    ] = True
    return long_format_feature_importance


# Apply SF to e^coeff mat data
#         pval_mat = pd.DataFrame(columns = mat.columns)
#         for class_lw in mat.index:
#             df_loadings = mat.loc[class_lw]
#             U = np.mean(df_loadings)
#             std = np.std(df_loadings)
#             med =  np.median(df_loadings)
#             mad = np.median(np.absolute(df_loadings - np.median(df_loadings)))
#             pvals = scipy.stats.norm.sf(df_loadings, loc=med, scale=1.96*U)


class estimate_important_features:  # This calculates feature effect sizes of the model
    def __init__(self, model, top_n):
        """
        General description.

        Parameters:

        Returns:

        """
        print("Estimating feature importance")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # get feature names
            try:
                model_features = list(itertools.chain(*list(model.features)))
            except:
                warnings.warn(
                    "no features recorded in data, naming features by position"
                )
                print(
                    "if low-dim lr was submitted, run linear decoding function to obtain true feature set"
                )
                model_features = list(range(0, model.coef_.shape[1]))
                model.features = model_features
            print("Calculating the Euler number to the power of coefficients")
            impt_ = pow(math.e, model.coef_)
            try:
                self.euler_pow_mat = pd.DataFrame(
                    impt_,
                    columns=list(itertools.chain(*list(model.features))),
                    index=list(model.classes_),
                )
            except:
                self.euler_pow_mat = pd.DataFrame(
                    impt_, columns=list(model.features), index=list(model.classes_)
                )
            self.top_n_features = pd.DataFrame(index=list(range(0, top_n)))
            # estimate per class feature importance

            print("Estimating feature importance for each class")
            mat = self.euler_pow_mat
            for class_pred_pos in list(range(0, len(mat.T.columns))):
                class_pred = list(mat.T.columns)[class_pred_pos]
                #     print(class_pred)
                temp_mat = pd.DataFrame(mat.T[class_pred])
                temp_mat["coef"] = model.coef_[class_pred_pos]
                temp_mat = temp_mat.sort_values(by=[class_pred], ascending=False)
                temp_mat = temp_mat.reset_index()
                temp_mat.columns = ["feature", "e^coef", "coef"]
                temp_mat = temp_mat[["feature", "e^coef", "coef"]]
                temp_mat.columns = str(class_pred) + "_" + temp_mat.columns
                self.top_n_features = pd.concat(
                    [self.top_n_features, temp_mat.head(top_n)],
                    join="inner",
                    ignore_index=False,
                    axis=1,
                )
                self.to_n_features_long = model_feature_sf(
                    long_format_features(self.top_n_features), "e^coef"
                )

    # plot class-wise features


def model_class_feature_plots(top_loadings, classes, comps, p_lim, max_len, title):
    """
    General description.

    Parameters:

    Returns:

    """
    import matplotlib.pyplot as plt

    for class_temp in classes:
        class_lw = class_temp
        long_format = top_loadings
        df_loadings = long_format[long_format["class"].isin([class_lw])]
        plt.hist(df_loadings[comps])

        if (
            len(
                (
                    (
                        df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
                    ).unique()
                )
            )
            < max_len
        ):
            for i in (
                df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
            ).unique():
                plt.axvline(x=i, color="red")
        elif (
            len(
                (
                    (
                        df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
                    ).unique()
                )
            )
            > max_len
        ):
            for i in (df_loadings[comps].nlargest(max_len)).unique():
                plt.axvline(x=i, color="red")
        med = np.median(df_loadings[comps])
        plt.axvline(x=med, color="blue")
        plt.xlabel("feature_importance", fontsize=12)
        plt.title(title + "_" + class_temp)
        # plt.axvline(x=med,color='pinkp_lim
        df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
        print(len(df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]))
        # Plot feature ranking
        if (
            len(
                (
                    (
                        df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
                    ).unique()
                )
            )
            < max_len
        ):
            plot_loading = pd.DataFrame(
                pd.DataFrame(
                    df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
                )
                .iloc[:, 0]
                .sort_values(ascending=False)
            )
        elif (
            len(
                (
                    (
                        df_loadings[comps][df_loadings[str(p_lim) + "_pval"] < 0.05]
                    ).unique()
                )
            )
            > max_len
        ):
            plot_loading = pd.DataFrame(
                pd.DataFrame(df_loadings[comps].nlargest(max_len))
                .iloc[:, 0]
                .sort_values(ascending=False)
            )
        table = plt.table(
            cellText=plot_loading.values,
            colWidths=[1] * len(plot_loading.columns),
            rowLabels=list(
                df_loadings["feature"][
                    df_loadings.index.isin(plot_loading.index)
                ].reindex(plot_loading.index)
            ),  # plot_loading.index,
            colLabels=plot_loading.columns,
            cellLoc="center",
            rowLoc="center",
            loc="right",
            bbox=[1.4, -0.05, 0.5, 1],
        )
        table.scale(1, 2)
        table.set_fontsize(10)


# this module calculates how well the features learnt by the model are distibuted amongst classes and the query data
def calculate_feature_distribution(adata, model, top_loadings, var="predicted"):
    """
    General description.

    Parameters:

    Returns:

    """
    adata_model = adata
    adata_model.obs["model_topn_feature_impact"] = 0

    # Loop
    long_format = top_loadings
    for class_temp in model.classes_:
        mask = adata_model.obs[var].isin([class_temp])
        adata_model[mask].X = sparse.csr_matrix(
            (
                np.multiply(
                    (adata_model[mask].X.todense()),
                    np.array(long_format.loc[class_temp]),
                )
            )
        )
        # Impact of top n features
        adata_model.obs.loc[mask, "model_topn_feature_impact"] = np.log(
            np.sum(
                np.multiply(
                    adata_model[mask][
                        :,
                        list(
                            long_format[long_format["class"].isin([class_temp])].feature
                        ),
                    ].X.todense(),
                    np.array(
                        long_format[long_format["class"].isin([class_temp])]["e^coef"]
                    ),
                ),
                axis=1,
            )
        )
    # Total impact of all model features
    adata_model.obs["model_class_impact_total"] = np.log(
        [
            item
            for sublist in (np.sum(adata_model.X, axis=1)).tolist()
            for item in sublist
        ]
    )

    return adata_model.obs[["model_topn_feature_impact", "model_class_impact_total"]]
