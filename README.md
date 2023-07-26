# scENTInEL: A Probabilistic Ensemble Framework for sc-Multiomic Data Transfer Learning

## Installation
To install directly from github run below in command line
```pip install git+https://github.com/Issacgoh/scENTInEL.git```

To clone and install:
```git clone https://github.com/Issacgoh/scENTInEL.git```
```cd ./scENTInEL```
```pip install .```

## Introduction

<<<<<<< HEAD
Scentinel is a probabilistic ensemble framework designed for mapping large single-cell (sc) multi-omic datasets using either transductive or inductive transfer learning. The framework operates on pre-integrated low-dimensional latent embeddings and offers multiple algorithmic modalities.

## Workflow

1. **Input and Dimensionality Reduction:** Scentinel initiates by taking an AnnData object as input. This package is agnostic to the type of dimensionality-reduced input. It can process outputs from PCA, VAE, linearly decoded VAE, NMF, or any other embedding structure. For use cases requiring inductive transfer learning generalizability, the package can directly handle raw data such as gene-expression, CITE-seq, or ATAC.

2. **Categorical Labels:** In addition to the AnnData object, Scentinel also expects a set of categorical labels to facilitate downstream ensemble learning processes.

3. **Core Algorithm:** The central engine of Scentinel is a tuned probabilistic Elastic Net classifier that delivers calibrated probabilities. Additionally, the package accommodates modalities based on XGBoost and SVM.

4. **Hyperparameter Optimization:** Scentinel carries out a Bayesian optimization step to tune the Elastic Net hyperparameters (alpha and L1-ratio). It utilizes negative Mean Absolute Error (neg_MAE) or log-loss as a loss function during this process. The Elastic Net objective function can be represented as:

    ```
    ```

    Here, `y` is the response vector, `X` is the predictor matrix, `w` are the coefficients to be estimated, `alpha` is the penalty term, and `l1_ratio` is the Elastic Net mixing parameter.

5. **Training the Classifier:** With the optimal set of hyperparameters, a probabilistic multinomial Elastic Net classifier is trained.

6. **Community Detection and Label Re-assignment:** Scentinel either takes pre-computed clusters or conducts community detection using the Leiden algorithm. Subsequently, labels are reassigned using a 68% confidence interval majority voting and frequency redistribution step. This mechanism enables the model to account for the local structure of the data.

7. **Modeling Uncertainty:** Scentinel optionally conducts a secondary ensemble step that models label uncertainty by considering local neighborhood structures. It offers options to model the likelihood of label consistency across neighbors using a Bayesian KNN approach or a GCN Gaussian process approach in an ensemble format.

8. **Harmonizing Labels:** This secondary step allows the harmonization of labels provided at different resolutions, removes labels unsupported by the data, or instills confidence in labels where a set of linearly separable complex decision boundaries can be learned.

9. **Inductive and Transductive Modes:** Scentinel can operate in two modes. In the inductive transfer learning mode, the model is trained to generalize to other datasets. This can be achieved by training the model directly on gene-expression/CITE/ATAC data or by training on a joint representation provided via architectural surgery (scARCHES). In the transductive transfer learning mode, the model is trained on a joint latent representation of the training and query data, thereby generalizing relationships from the training data to the entire dataset, followed by uncertainty modeling using a Gaussian Process.
