# scENTInEL: A Probabilistic Ensemble Framework for sc-Multiomic Data Transfer Learning

## Installation
To install directly from github run below in command line

```pip install git+https://github.com/Issacgoh/scENTInEL.git```

To clone and install:

```git clone https://github.com/Issacgoh/scENTInEL.git```

```cd ./scENTInEL```

```pip install .```

## Introduction

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

7. **Modeling Uncertainty:** Scentinel optionally conducts a secondary ensemble step that models label uncertainty by considering local neighborhood structures. It offers options to model the likelihood of label consistency across neighbors using a Bayesian KNN approach, a GCN Gaussian process approach in an ensemble format via predicted label entropy in each neighborhood.

8. **Harmonizing Labels:** This secondary step allows the harmonization of labels provided at different resolutions, removes labels unsupported by the data, or instills confidence in labels where a set of linearly separable complex decision boundaries can be learned. We can alternatively use a heirarichal model trained on the predicted probabilities from XGboost or LR and determine a dynamic branch cutoff. 

9. **Inductive and Transductive Modes:** Scentinel can operate in two modes. In the inductive transfer learning mode, the model is trained to generalize to other datasets. This can be achieved by training the model directly on gene-expression/CITE/ATAC data or by training on a joint representation provided via architectural surgery (scARCHES). In the transductive transfer learning mode, the model is trained on a joint latent representation of the training and query data, thereby generalizing relationships from the training data to the entire dataset, followed by uncertainty modeling using a Gaussian Process.


## Notes for approaches:
## Mini-batch SGD PageRank with Laplacian Matrix

PageRank is an iterative method to compute the importance of each node in a graph. The original idea, pioneered by Google founders Larry Page and Sergey Brin, was that the importance of a webpage is determined by the importance of the pages that link to it.

### Classic PageRank Formula

Mathematically, the classic PageRank equation for a node \(i\) is:

\[
PR(i) = (1 - d) + d 	imes \sum_{j \in M(i)} rac{PR(j)}{L(j)}
\]

Where:
- \( PR(i) \) is the PageRank of node \(i\).
- \( d \) is the damping factor, typically set to around 0.85.
- \( M(i) \) is the set of nodes that link to node \(i\).
- \( L(j) \) is the number of outbound links from node \(j\).

### Our Implementation

Our implementation integrates the use of the Laplacian matrix and a mini-batch stochastic gradient descent (SGD) approach to optimize the PageRank values iteratively.

Pros:
Efficiency: When working with large graphs, updating the scores of all nodes in each iteration can be computationally expensive. By using mini-batches, the algorithm can make progress towards convergence using only a fraction of the nodes in each iteration.
Escape Local Minima: The randomness introduced by mini-batches can help the algorithm escape local minima.

Cons:
No Guarantee of Visiting All Nodes: As you correctly pointed out, there's no guarantee that all nodes will be visited. Some nodes might be visited multiple times while others might not be visited at all.
Convergence Stability: The stochastic nature can lead to more noisy updates, which might affect the stability of convergence.

it's worth noting that the strength of SGD, especially with mini-batches, is that it often works well in practice, even without visiting all data points (or nodes, in this case), because the random subsets often provide a good enough approximation for the entire dataset.
 
#### Laplacian Approach

The normalized Laplacian matrix captures the structure of the graph in a way that nodes are penalized for having a high degree, considering both local and global structures. The formula to compute the normalized matrix is:

\[
L_{	ext{normalized}} = D^{-rac{1}{2}} L D^{-rac{1}{2}}
\]

Where \(L\) is the Laplacian matrix and \(D\) is the diagonal matrix of node degrees. By penalizing nodes with a higher degree, the normalized Laplacian offers a balance between local and global importance in the graph.

#### Mini-batch SGD Iterations

1. **Initialization**:
   - A random rank vector `v` of size \(N\) (number of nodes) is initialized and normalized.
   
2. **Iteration**:
   For each iteration:
   - A learning rate is calculated, which decays over iterations.
   - A subset (mini-batch) of node indices is randomly selected.
   - For the selected mini-batch, the PageRank update is applied using the Laplacian matrix.
   - The PageRank values for the nodes in the mini-batch are updated.
   - Convergence is checked by comparing the updated PageRank values to the previous iteration's values.

3. **Output**:
   - The function returns the estimated PageRank values for all nodes and the L2 norm of the difference between successive estimates for each iteration.

### Benefits

- **Efficiency with Large Graphs**: Mini-batch SGD is computationally efficient, especially for large graphs.
- **Convergence**: Adjusting the learning rate and using a decay factor aids in convergence.
- **Flexibility**: Provides flexibility in terms of batch size and learning rate.
- **Capturing Local Structures**: By using mini-batches and the Laplacian matrix, the algorithm captures both local and global structures in the graph.

By integrating the Laplacian matrix approach with PageRank, we aim to identify influential nodes in the graph, focusing on both their local and global importance.




