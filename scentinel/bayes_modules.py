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

import gc
import math
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.stats import entropy
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # Import tqdm for progress bars

sns.set_theme(color_codes=True)


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
    y_true = pd.get_dummies(df[true_label])
    y_pred = df[pred_columns]

    # Get all unique classes from the true labels
    unique_classes = np.sort(df[true_label].unique())
    # Convert categorical true labels to one-hot encoding
    y_true = pd.get_dummies(df[true_label], columns=unique_classes)
    # Make sure y_pred has columns for all classes in y_true, fill missing with zeros
    y_pred = df[pred_columns].reindex(columns=unique_classes, fill_value=0)
    loss = log_loss(np.array(y_true), np.array(y_pred))
    for label in range(y_true.shape[1]):
        log_loss_label = log_loss(
            np.array(y_true)[:, label], np.array(y_pred)[:, label]
        )
        log_losses[list(y_true.columns)[label]] = log_loss_label
    weights = 1 / np.array(list(log_losses.values()))
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
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    #     r2=metrics.r2_score(y_true, y_pred)
    print("Cross entropy loss: ", round(loss, 4))
    print("mean_squared_log_error: ", round(mean_squared_log_error, 4))
    print("MAE: ", round(mean_absolute_error, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(np.sqrt(mse), 4))
    print("label Cross entropy loss: ")
    print(log_losses)
    return loss, log_losses, weights


def V0_3_empirical_bayes_balanced_stratified_KNN_sampling(
    adata,
    feat_use,
    knn_key,
    sampling_rate=0.1,
    iterations=1,
    representation_priority=0.9,
    equal_allocation=False,
    replace=True,
    weight_penalty="connectivity_ratio",
    **kwargs,
):
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())
    #     print(locals())

    if equal_allocation:
        print(
            "You are using an equal allocation mode of sampling, be warned that this can cause errors if the smaller populations are insufficient in number, consider replace == True"
        )

    if replace == True:
        print(
            "You are using sampling with replacement, this allows the model to create clones of cells"
        )

    if representation_priority < 0.6:
        print(
            "warning: you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )
        warnings.warn(
            "warning you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )

    # Convert string labels to integer labels
    unique_labels, indices = np.unique(adata.obs[feat_use], return_inverse=True)
    adata.obs["int.labels"] = indices

    # Calculate frequencies (prior probabilities)
    label_counts = np.bincount(indices)
    frequencies = label_counts / label_counts.sum()

    # Create a dictionary where keys are label indices and values are their frequencies (prior probabilities)
    prior_distribution = dict(zip(range(len(unique_labels)), frequencies))

    neighborhood_matrix = adata.obsp[adata.uns[knn_key]["connectivities_key"]]

    # Initialize label probabilities with prior distribution
    label_probs = prior_distribution.copy()

    # Get indices for each label
    label_indices = {
        label: np.where(adata.obs["int.labels"] == label)[0]
        for label in range(len(unique_labels))
    }

    # Calculate total sample size and sample size per label for equal allocation
    total_sample_size = int(sampling_rate * adata.shape[0])
    sample_size_per_label = total_sample_size // len(unique_labels)

    if weight_penalty == "entropy_distance":
        print(
            "Using distance-entropy penalisation weights, this module is multi-threaded and quite compute intensive. If facing issues, use connectivity_ratio instead"
        )
        # Calculate entropy for each neighborhood in advance
        all_labels = adata.obs["int.labels"].values
        neighborhood_indices = [
            np.nonzero(neighborhood_matrix[idx])[1] for idx in range(adata.shape[0])
        ]
        # Calculate entropy for each cell in parallel
        import multiprocessing

        with multiprocessing.Pool() as pool:
            neighborhood_entropies = np.array(
                list(
                    pool.map(
                        calculate_entropy,
                        [(all_labels, idx) for idx in neighborhood_indices],
                    )
                )
            )

    # Create a dictionary to store the neighborhood entropy for each label at each iteration
    neighborhood_entropies_iter = {label: [] for label in range(len(unique_labels))}
    sampling_probabilities_over_iterations = np.zeros((iterations, len(unique_labels)))
    for _ in range(iterations):
        print("Iteration: {}".format(_))
        # Stratified sampling within each neighborhood for each label
        all_weights = []
        all_indices = []
        for label in label_indices.keys():
            indices = label_indices[label]
            other_indices = [
                idx
                for key, indices in label_indices.items()
                if key != label
                for idx in indices
            ]

            neighborhoods = neighborhood_matrix[
                indices
            ]  # select neighborhoods for the current label

            # Here we determine the liklighood that a sampled cell forms consistant neigthborhood
            # We take the sum of weights of the same labels within each neighborhood to represent the liklihood of a cell forming tight communities (px)
            # We divide p by the sum of weights of different labels in the same neighborhood to represent the liklihood of there being more than one state sharing the neighborhood. (nx)
            # We take the ratio of (px/nx) to represent the liklihood that a label represents consistant neighborhoods that are sufficiently independent
            # e.g The sum of weights of cells with different labels in a neighborhood relative to our observed cell should be low if the label for the cell is good. thus the ratio should be high.
            if weight_penalty == "connectivity_ratio":
                same_label_mask = np.array(
                    adata.obs["int.labels"][indices] == label, dtype=int
                )  # get mask for same-label cells
                same_label_mask = scipy.sparse.diags(
                    same_label_mask
                )  # convert to diagonal matrix for multiplication
                same_label_neighborhoods = neighborhoods[
                    :, indices
                ]  # get neighborhoods of same-label cells
                different_label_neighborhoods = neighborhoods[
                    :, other_indices
                ]  # get neighborhoods of different-label cells
                same_label_weights = np.array(
                    same_label_neighborhoods.sum(axis=1)
                ).ravel()
                different_label_weights = np.array(
                    different_label_neighborhoods.sum(axis=1)
                ).ravel()
                # we now compute a hybrid weighting scheme, where cells with more defined neigthborhood structure are prioritised. Howevever we do not want this over penalise, so we recover underrepresented cells with a inverse weighting parameter
                # Calculate the ratio of same-label weights to different-label weights
                # Add a small constant in the denominator to avoid division by zero
                weights = (same_label_weights) / (
                    different_label_weights + 1e-8
                )  # if same label sum distances are 0, do not consider this cell
                weights = weights / np.sum(
                    weights
                )  # this normnalisation means that every label has normalised set of weights to bias selection

            # Here we determine the liklighood that a sampled cell forms consistant neigthborhood using the mean distance to all neighbors * by the entropy of the neighborhood

            if weight_penalty == "entropy_distance":
                # We take neighborhoods that share the same labels and compute the sum of weights between same labels
                #             np.array(neighborhoods[:, indices].sum(axis=1)).ravel()
                same_label_mask = np.array(
                    adata.obs["int.labels"][indices] == label, dtype=int
                )  # get mask for same-label cells
                same_label_mask = scipy.sparse.diags(
                    same_label_mask
                )  # convert to diagonal matrix for multiplication
                same_label_neighborhoods = neighborhoods[:, indices]
                weights = (
                    np.array(same_label_neighborhoods.sum(axis=1)).ravel() + 1e-8
                )  # We take the sum of weights to all neighbors of the same label here
                # 1/weights give us the inverse where big weights are big distances
                weights *= 1 / (
                    neighborhood_entropies[indices] + 1e-8
                )  # use pre-computed entropies
                #             weights = weights
                weights = weights / np.sum(
                    weights
                )  # this normnalisation means that every label has normalised set of weights to bias selection

            # Update weights based on representation priority and label probabilities
            # This should be a combination of the neighborhood-based weights and the label probability-based weights
            if representation_priority != 0:
                weights = weights * (
                    (1 / (label_probs[label] + 1e-8)) ** representation_priority
                )
            else:
                weights = weights * (1 / (label_probs[label] + 1e-8))
            # weights = np.array(weights) / np.sum(weights)  # normalization to probabilities
            all_weights.extend(weights)
            all_indices.extend(indices)

        all_weights = np.array(all_weights) / np.sum(
            all_weights
        )  # normalization to probabilities
        sample_indices = np.random.choice(
            all_indices, size=total_sample_size, replace=replace, p=all_weights
        )

        # Update label probabilities based on the observed sample
        sample_labels = adata.obs["int.labels"][sample_indices]
        label_counts = np.bincount(sample_labels, minlength=len(unique_labels))
        label_probs = dict(
            zip(range(len(unique_labels)), label_counts / label_counts.sum() + 1e-8)
        )
        # Store the sampling probabilities for this iteration
        sampling_probabilities_over_iterations[_, :] = np.array(
            list(label_probs.values())
        )

        # Calculate the entropy for the sampled cells
        for label in label_indices.keys():
            # Get the indices of the sampled cells for the current label
            sampled_indices = [
                idx for idx in sample_indices if adata.obs["int.labels"][idx] == label
            ]
            if sampled_indices:
                # Get neighborhoods of the sampled cells
                same_label_neighborhoods = neighborhood_matrix[sampled_indices]
                # Get the indices of the connected cells
                connected_indices = same_label_neighborhoods.nonzero()[1]
                # Get the labels of the connected cells
                connected_labels = adata.obs["int.labels"].values[connected_indices]
                # Calculate the entropy for the current label
                label_counts = np.bincount(
                    connected_labels, minlength=len(unique_labels)
                )
                probabilities = label_counts / len(connected_indices)
                entropy_val = entropy(probabilities)
                neighborhood_entropies_iter[label].append(entropy_val)
            else:
                neighborhood_entropies_iter[label].append(None)

    average_sampling_probabilities = sampling_probabilities_over_iterations.mean(axis=0)
    updated_label_probs = dict(
        zip(range(len(unique_labels)), average_sampling_probabilities)
    )

    # Final stratified sampling using the last label_probs
    label_sample_sizes = {
        label: int(updated_label_probs[label] * total_sample_size)
        for label in range(len(unique_labels))
    }
    # Adjust sample sizes so total equals 'total_sample_size'
    difference = total_sample_size - sum(label_sample_sizes.values())
    label_sample_sizes[0] += difference  # adjust the first label for simplicity
    final_sample_indices = []

    if equal_allocation == True:
        label_sample_sizes = {
            label: sample_size_per_label for label in range(len(unique_labels))
        }

    for label, sample_size in label_sample_sizes.items():
        indices = label_indices[label]
        neighborhoods = neighborhood_matrix[indices][
            :, indices
        ]  # select neighborhoods for the current label

        same_label_mask = np.array(
            adata.obs["int.labels"][indices] == label, dtype=int
        )  # get mask for same-label cells
        same_label_mask = scipy.sparse.diags(
            same_label_mask
        )  # convert to diagonal matrix for multiplication

        same_label_neighborhoods = (
            same_label_mask @ neighborhoods @ same_label_mask
        )  # get neighborhoods of same-label cells
        different_label_neighborhoods = (
            neighborhoods - same_label_neighborhoods
        )  # get neighborhoods of different-label cells

        same_label_weights = np.array(same_label_neighborhoods.sum(axis=1)).ravel()
        different_label_weights = np.array(
            different_label_neighborhoods.sum(axis=1)
        ).ravel()

        # Calculate the ratio of same-label weights to different-label weights
        # Add a small constant in the denominator to avoid division by zero
        # weights = same_label_weights / (different_label_weights + 1e-8)
        #     weights = weights / weights.sum()  # normalization to probabilities
        specific_weights = np.array(all_weights[indices]) / np.sum(all_weights[indices])
        try:
            sampled_indices = np.random.choice(
                indices, size=sample_size, replace=replace, p=specific_weights
            )
        except:
            print(
                "Warning -- sampling for {} without replacement failed, defaulting to taking all cells in this category".format(
                    label
                )
            )
            sampled_indices = np.random.choice(
                indices, size=len(indices), replace=False, p=specific_weights
            )
        final_sample_indices.extend(sampled_indices)
    adata_samp = adata[final_sample_indices, :]
    all_weights

    # plot entropy change per iteration
    # Calculate the number of columns for the legend
    ncol = math.ceil(
        len(unique_labels) / 20
    )  # Adjust the denominator to control the number of legend entries per column

    # Create a figure and an axes object
    fig, ax = plt.subplots(
        figsize=(5 + ncol, 5)
    )  # Adjust as needed. The width of the axes object will be always 5.

    # Compute the initial entropies for the whole dataset
    initial_entropies = {}
    for label in label_indices.keys():
        indices = label_indices[label]
        same_label_neighborhoods = neighborhood_matrix[indices]
        connected_indices = same_label_neighborhoods.nonzero()[1]
        connected_labels = adata.obs["int.labels"].values[connected_indices]
        label_counts = np.bincount(connected_labels, minlength=len(unique_labels))
        probabilities = label_counts / len(connected_indices)
        entropy_val = entropy(probabilities)
        initial_entropies[label] = entropy_val

    # Plot the change in neighborhood entropy over iterations for each label
    for label, entropies in neighborhood_entropies_iter.items():
        # Prepend the initial entropy to the list of entropies
        all_entropies = [initial_entropies[label]] + entropies
        ax.plot(range(len(all_entropies)), all_entropies, label=unique_labels[label])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Scaled Neighborhood Entropy")
    # Set the y-axis to logarithmic scale
    ax.set_yscale("log")
    # Place the legend outside the plot, scaled with the height of the plot and spread into columns
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0, ncol=ncol)
    plt.tight_layout()
    plt.show()
    return adata_samp, final_sample_indices


def update_connectivity_matrix_in_chunks_v0_1_0(
    KNN_main, updates_dict, chunk_size=1000
):
    """
    Updates the connectivity matrix in chunks for memory efficiency with a progress bar.

    Args:
    - KNN_main: The main connectivity matrix.
    - updates_dict: Dictionary containing updates for each epoch.
    - chunk_size: Size of each chunk for updating the matrix.

    Returns:
    An updated connectivity matrix.
    """
    print("Updating connectivity matrix in chunks")
    # Determine the latest epoch for each vertex
    latest_updates = {}
    for epoch, (indices, KNN_hop) in updates_dict.items():
        for idx, original_idx in enumerate(indices):
            latest_updates[original_idx] = KNN_hop[idx, :]

    all_indices = sorted(list(latest_updates.keys()))
    total_indices = len(all_indices)

    # Update the main matrix in chunks with tqdm progress bar
    for i in tqdm(
        range(0, total_indices, chunk_size),
        total=len(range(0, total_indices, chunk_size)),
    ):
        chunk_indices = all_indices[i : i + chunk_size]
        for idx in chunk_indices:
            KNN_main[idx, :] = latest_updates[idx]

    # Ensure the matrix is symmetrical
    KNN_main = (KNN_main + KNN_main.transpose()).tocsr()
    KNN_main.data = np.where(KNN_main.data > 0, 1, 0)

    return KNN_main


def update_connectivity_matrix_in_chunks_v0_1_0(
    KNN_main, updates_dict, chunk_size=1000
):
    """
    Updates the connectivity matrix in chunks for memory efficiency with a progress bar.

    Args:
    - KNN_main: The main connectivity matrix.
    - updates_dict: Dictionary containing updates for each epoch.
    - chunk_size: Size of each chunk for updating the matrix.

    Returns:
    An updated connectivity matrix.
    """
    print("Updating connectivity matrix in chunks")

    # Convert the main matrix to LIL format for efficient row-wise operations
    KNN_main_lil = KNN_main.tolil()

    # Process updates more efficiently
    all_updates = []
    for epoch, (indices, KNN_hop) in updates_dict.items():
        for idx, original_idx in enumerate(indices):
            all_updates.append((original_idx, KNN_hop[idx, :]))

    # Sort updates by index for efficient batch updating
    all_updates.sort(key=lambda x: x[0])

    # Update the main matrix in chunks
    for i in tqdm(
        range(0, len(all_updates), chunk_size), desc="Updating connectivity matrix"
    ):
        # Process each chunk
        for original_idx, update in all_updates[i : i + chunk_size]:
            KNN_main_lil[original_idx, :] = update

        # Manual memory management
        gc.collect()

    # Convert back to CSR format after updates
    KNN_main_updated = KNN_main_lil.tocsr()

    # Symmetrize the matrix efficiently
    KNN_main_sym = KNN_main_updated.maximum(KNN_main_updated.transpose())
    KNN_main_sym.data = np.where(KNN_main_sym.data > 0, 1, 0)

    return KNN_main_sym


# Update the main function to use the corrected update function
def expand_neighborhoods_chunked_v0_1_0(adata, adata_samp, param_set):
    KNN_main = adata.obsp["connectivities"]
    updates_dict = {}  # Store the updates for each epoch
    results_dict = {}  # Final results dictionary

    epoch = 0
    hop_v_indices = [1]  # Initialize with a non-empty list to start the loop.
    samp_indices = np.where(adata.obs.index.isin(adata_samp.obs.index))[0]

    while epoch <= param_set["epoch"] and len(hop_v_indices) > 0:
        print(f"Epoch: {epoch}")

        # Extract submatrix based on sampled data
        KNN_tmp = KNN_main[adata.obs.index.isin(adata_samp.obs.index)]
        KNN_tmp.data = np.where(KNN_tmp.data > 0, 1, 0)

        # Find indices not connected to any sampled node
        if epoch == 0:
            indices = np.where(KNN_tmp.sum(axis=0) <= param_set["alpha"])[1]
        else:
            indices = hop_v_indices

        KNN_hop = KNN_main[indices]
        KNN_hop = KNN_hop * KNN_main

        # Convert KNN_hop to a temporary binary neighborhood matrix
        KNN_hop_tmp = KNN_hop[:, samp_indices].copy()
        KNN_hop_tmp.data = np.where(KNN_hop_tmp.data > 0, 1, 0)

        hop_v_indices = np.where(KNN_hop_tmp.sum(axis=1) <= param_set["alpha"])[0]
        updates_dict[epoch] = (indices, KNN_hop)

        epoch += 1

    # Reconstruct the main matrix using the chunked update function
    KNN_updated = update_connectivity_matrix_in_chunks(KNN_main, updates_dict)

    results_dict["main_matrix"] = KNN_updated
    results_dict["updates"] = updates_dict
    sp_v_indices = np.where(
        results_dict["main_matrix"].sum(axis=0) <= param_set["alpha"]
    )[1]
    print("Remaining unconnected node count is: {}".format(len(sp_v_indices)))

    return results_dict


# results = expand_neighborhoods_chunked_corrected(adata, adata_samp, param_set)

#######################


def apply_adaptive_gaussian_kernel(
    KNN, anchor_indices, adp_variance=2.0, adp_threshold=0.2, **kwargs
):
    """
    Apply an adaptive Gaussian kernel to the connectivity matrix.

    Args:
    - KNN: The connectivity matrix.
    - anchor_indices: Indices of anchor nodes to preserve connections.
    - variance: Variance parameter for the Gaussian kernel.
    - threshold: Threshold to prune connections.

    Returns:
    A pruned connectivity matrix with Gaussian kernel applied.
    """
    if kwargs:
        locals().update(kwargs)
        kwargs.update(locals())

    print("Applying Adaptive gaussian kernel to prune connections")
    # Convert to COO format for easier element-wise operations
    KNN_dyn = KNN.copy()

    # Convert KNN.data to float64 to ensure compatibility with Gaussian kernel values
    KNN_dyn.data = KNN_dyn.data.astype(np.float64)

    # Calculate Gaussian kernel values
    gaussian_kernel_values = np.exp(-KNN_dyn.data**2 / (2.0 * adp_variance))

    # Apply kernel to the connectivity matrix
    KNN_dyn.data *= gaussian_kernel_values

    # Create a mask to identify anchor connections
    anchor_mask = np.isin(KNN_dyn.indices, anchor_indices)

    # Prune connections below the threshold, except for anchor connections
    KNN.data[~anchor_mask] = np.where(
        KNN_dyn.data[~anchor_mask] > adp_threshold, KNN.data[~anchor_mask], 0
    )
    # KNN_dyn.data[~anchor_mask] = np.where(KNN_dyn.data[~anchor_mask] > adp_threshold, KNN_dyn.data[~anchor_mask], 0)

    # Eliminate zero entries and convert back to CSR format
    KNN.eliminate_zeros()

    # Convert back to int64 format
    KNN = KNN.tocsr()
    # pruned_to_zero_mask = (KNN_dyn != KNN).multiply(KNN_dyn == 0)
    # KNN[pruned_to_zero_mask] = 0

    return KNN


def v0_2_1_process_chunk(start_idx, end_idx, updates, KNN_main_format):
    KNN_main_local = KNN_main_format.copy()
    for idx in range(start_idx, end_idx):
        original_idx, update = updates[idx]
        KNN_main_local[original_idx, :] = update
    return KNN_main_local


def v0_2_1_update_connectivity_matrix_in_chunks(
    KNN_main, updates_dict, chunk_size=10000, n_jobs=4
):
    """
    Updates the connectivity matrix in chunks for memory efficiency with a progress bar.

    Args:
    - KNN_main: The main connectivity matrix.
    - updates_dict: Dictionary containing updates for each epoch.
    - chunk_size: Size of each chunk for updating the matrix.

    Returns:
    An updated connectivity matrix.
    """
    print("Updating connectivity matrix in chunks")

    # Convert the main matrix to LIL format for efficient row-wise operations
    KNN_main_lil = KNN_main.tolil()

    if n_jobs > 1:
        # Check the maximum number of available cores and adjust n_jobs if necessary
        max_cores = cpu_count()
        print("{} compute cores available".format(max_cores))
        n_jobs = min(n_jobs, max_cores)

        all_updates = [
            (original_idx, KNN_hop[idx, :])
            for epoch, (indices, KNN_hop) in updates_dict.items()
            for idx, original_idx in enumerate(indices)
        ]
        all_updates.sort(key=lambda x: x[0])

        print("Using parallel processing with {} jobs".format(n_jobs))
        chunks = [
            (i, min(i + chunk_size, len(all_updates)), all_updates, KNN_main_lil)
            for i in range(0, len(all_updates), chunk_size)
        ]
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk)(start, end, all_updates, KNN_main_lil)
            for start, end, _, _ in chunks
        )
        print("Jobs returned")
        for chunk in processed_chunks:
            KNN_main_lil = KNN_main_lil.maximum(chunk)
    else:
        print("Using non-parallel processing")
        # Process updates more efficiently
        all_updates = []
        for epoch, (indices, KNN_hop) in updates_dict.items():
            for idx, original_idx in enumerate(indices):
                all_updates.append((original_idx, KNN_hop[idx, :]))

        # Sort updates by index for efficient batch updating
        all_updates.sort(key=lambda x: x[0])

        # Update the main matrix in chunks
        for i in tqdm(
            range(0, len(all_updates), chunk_size), desc="Updating connectivity matrix"
        ):
            # Process each chunk
            for original_idx, update in all_updates[i : i + chunk_size]:
                KNN_main_lil[original_idx, :] = update

        # Manual memory management
        gc.collect()

    print("Reconstructing connectivity matrix")
    # Convert back to CSR format after updates
    KNN_main_updated = KNN_main_lil.tocsr()

    # Symmetrize the matrix efficiently
    # KNN_main_sym = KNN_main_updated.maximum(KNN_main_updated.transpose())
    KNN_main_sym = KNN_main_updated
    # KNN_main_sym.data = np.where(KNN_main_sym.data > 0, 1, 0)

    return KNN_main_sym


def process_chunk(compiled_updates, chunk, force_symmetry, KNN_main_shape):
    """
    Processes a chunk of updates and constructs a COO matrix representing the updates for this chunk.

    Parameters:
    - compiled_updates (dict): A dictionary containing the latest updates for each unique index.
    - chunk (list): A list of original indices that represent a subset of updates to be processed.
    - force_symmetry (bool): Whether to force symmetry in the updates.
    - KNN_main_shape (tuple): The shape of the main KNN matrix.

    Returns:
    - coo_matrix: A COO matrix representing the updates for the provided chunk.
    """
    row_updates, col_updates, data_updates = [], [], []
    for original_idx in chunk:
        update_coo = compiled_updates[original_idx]
        row_updates.extend([original_idx] * len(update_coo.data))
        col_updates.extend(update_coo.col)
        data_updates.extend(update_coo.data)
        if force_symmetry:
            row_updates.extend(update_coo.col)
            col_updates.extend([original_idx] * len(update_coo.data))
            data_updates.extend(update_coo.data)
    return coo_matrix((data_updates, (row_updates, col_updates)), shape=KNN_main_shape)


def update_connectivity_matrix_in_chunks(
    KNN_main, updates_dict, n_jobs=4, force_symmetry=False, **kwargs
):
    """
    Compiles updates from an updates dictionary and applies them to the KNN_main CSR matrix. Supports both
    parallel and non-parallel execution modes and can optionally force symmetry in the updates.

    Parameters:
    - KNN_main (csr_matrix): The main KNN matrix to which updates will be applied.
    - updates_dict (dict): A dictionary containing updates, structured as {epoch: (indices, KNN_hop)}.
    - n_jobs (int): The number of parallel jobs to use for processing updates.
    - force_symmetry (bool): If True, forces symmetry in the updates.

    Returns:
    - csr_matrix: The updated KNN_main matrix.
    """
    if kwargs:
        locals().update(kwargs)
        kwargs.update(locals())
    version = "V0.2.2"
    print("You are running {} module".format(version))

    compiled_updates = {}
    for epoch, (indices, KNN_hop) in sorted(updates_dict.items(), reverse=True):
        for idx, original_idx in enumerate(indices):
            if original_idx not in compiled_updates:
                compiled_updates[original_idx] = KNN_hop[idx, :].tocoo()

    if force_symmetry:
        print("module to force symmetry active")

    if n_jobs <= 1:
        print("Non-parallel update module chosen")
        row_updates, col_updates, data_updates = [], [], []
        pbar = tqdm(total=len(compiled_updates), desc="Processing updates")
        for original_idx, update_coo in compiled_updates.items():
            row_updates.extend([original_idx] * len(update_coo.data))
            col_updates.extend(update_coo.col)
            data_updates.extend(update_coo.data)
            if force_symmetry:
                row_updates.extend(update_coo.col)
                col_updates.extend([original_idx] * len(update_coo.data))
                data_updates.extend(update_coo.data)
            pbar.update(1)
        pbar.close()
        updates_coo = coo_matrix(
            (data_updates, (row_updates, col_updates)), shape=KNN_main.shape
        )
    else:
        print("Parallel update module available, proceeding")
        max_cores = cpu_count()
        n_jobs = min(n_jobs, max_cores)
        print(f"{max_cores} compute cores available, distributing {n_jobs} jobs")
        chunks = np.array_split(list(compiled_updates.keys()), n_jobs)
        with Pool(processes=n_jobs) as pool:
            update_results = list(
                tqdm(
                    pool.imap(
                        partial(
                            process_chunk,
                            compiled_updates,
                            force_symmetry=force_symmetry,
                            KNN_main_shape=KNN_main.shape,
                        ),
                        chunks,
                    ),
                    total=len(chunks),
                    desc="Processing chunks in parallel",
                )
            )
        updates_coo = sum(update_results)

    KNN_main += updates_coo.tocsr()
    return KNN_main


def expand_neighborhoods_chunked(
    adata, adata_samp, n_jobs=1, adaptive_prune=False, **kwargs
):
    """
    Expands the neighborhoods in the connectivity matrix of anndata object 'adata'
    by dynamically hopping to neighboring nodes, updating the connectivity matrix in chunks.

    Args:
    - adata: Anndata object containing the main dataset.
    - adata_samp: Anndata object containing the sampled dataset.
    - n_jobs: Number of parallel jobs to use for updating the connectivity matrix.
    - **kwargs: Additional keyword arguments including:
        - 'epoch': Maximum number of iterations for neighborhood expansion.
        - 'alpha': Threshold to determine connected nodes.

    Returns:
    A dictionary containing:
    - 'main_matrix': The updated main connectivity matrix after neighborhood expansion.
    - 'updates': A dictionary storing the updates for each epoch, including the updated matrices.

    This function iteratively expands the neighborhoods in the connectivity matrix. For each epoch,
    it identifies new nodes to connect based on a threshold 'alpha' and updates the connectivity
    matrix. Each update is followed by a pruning stage via an adaptive gaussian kernel. The process continues until the specified number of epochs is reached or no new nodes
    are identified. The 'updates' in the results dictionary includes the details of updates made
    in each epoch, providing insight into the dynamic expansion process.
    """

    # unpack kwargs
    if kwargs:
        locals().update(kwargs)
    kwargs.update(locals())

    KNN_main = adata.obsp[adata.uns[kwargs["knn_key"]]["connectivities_key"]]
    updates_dict = {}  # Store the updates for each epoch
    results_dict = {}  # Final results dictionary

    epoch = 0
    hop_v_indices = [1]  # Initialize with a non-empty list to start the loop.
    samp_indices = np.where(adata.obs.index.isin(adata_samp.obs.index))[0]

    if kwargs["alpha"] == 0:
        warnings.warn("Alpha value set to <1")
        print(
            "Alpha is set to 0, this means that no expansion towards anchor states will occur, we will default this value to 1"
        )
        kwargs["alpha"] = 1

    while epoch <= kwargs["epoch"] and len(hop_v_indices) > 0:
        print(f"Epoch: {epoch}")

        # Extract submatrix based on sampled data
        KNN_tmp = KNN_main[adata.obs.index.isin(adata_samp.obs.index)]
        KNN_tmp.data = np.where(KNN_tmp.data > 0, 1, 0)

        # Find indices not connected to any sampled node
        if epoch == 0:
            indices = np.where(KNN_tmp.sum(axis=0) < kwargs["alpha"])[1]  # axis 0 here
        else:
            indices = hop_v_indices

        KNN_hop = KNN_main[indices]
        KNN_hop = KNN_hop * KNN_main

        # Convert KNN_hop to a temporary binary neighborhood matrix
        KNN_hop_tmp = KNN_hop[:, samp_indices].copy()
        KNN_hop_tmp.data = np.where(KNN_hop_tmp.data > 0, 1, 0)

        hop_v_indices = np.where(KNN_hop_tmp.sum(axis=1) < kwargs["alpha"])[
            0
        ]  # axis 1 here
        updates_dict[epoch] = (indices, KNN_hop)

        print(
            "End of epoch {} Unconnected node count is: {}".format(
                epoch, len(hop_v_indices)
            )
        )
        epoch += 1

    KNN_updated = update_connectivity_matrix_in_chunks(KNN_main, updates_dict, **kwargs)

    if adaptive_prune == True:
        KNN_updated = apply_adaptive_gaussian_kernel(
            KNN_updated,
            np.where(adata.obs.index.isin(adata_samp.obs.index))[0],
            adp_variance=1.0,
            adp_threshold=0.1,
            **kwargs,
        )
    results_dict = {"main_matrix": KNN_updated, "updates": updates_dict}
    sp_v_indices = np.where(results_dict["main_matrix"].sum(axis=0) < kwargs["alpha"])[
        1
    ]
    print("Remaining unconnected node count is: {}".format(len(sp_v_indices)))

    return results_dict


# Example usage
# result = expand_neighborhoods_chunked(adata, adata_samp, param_set, n_jobs=4)


def pagerank(M, num_iterations=100, d=0.85, tolerance=1e-6):
    """
    Calculate the PageRank of each node in a graph.

    Parameters:
    M (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
    num_iterations (int): The maximum number of iterations to perform.
    d (float): The damping factor.
    tolerance (float): The tolerance for convergence.

    Returns:
    numpy.ndarray: The PageRank score for each node in the graph.
    """
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    damping = ((1 - d) / N) * np.ones((N, 1))
    last_v = np.ones((N, 1)) * np.inf
    l2_dic = {}
    for _ in range(num_iterations):
        last_v = v
        v = d * M @ v + damping
        l2_norm = np.linalg.norm(v - last_v)
        l2_dic[_] = l2_norm
        if l2_norm < tolerance:
            print("Converged at iteration {}".format(_))
            break

    plt.figure(figsize=(10, 6))
    plt.plot(list(l2_dic.keys()), list(l2_dic.values()))
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm")
    plt.title("Convergence of PageRank")
    plt.grid(True)
    plt.show()

    return v, l2_dic


def SGDpagerank_v0_1_0(
    M,
    init_vect=None,
    num_iterations=1000,
    mini_batch_size=1000,
    initial_learning_rate=0.85,
    tolerance=1e-5,
    d=0.85,
    full_batch_update_iters=100,
    dip_window=5,
    plateau_iterations=5,
    sampling_method="probability_based",
    **kwargs,
):
    """
    Calculate the PageRank of each node in a graph using a mini-batch SGD approach.

    Parameters:
    - M (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
    - num_iterations (int): The maximum number of iterations to perform.
    - mini_batch_size (int): Number of nodes to sample in each iteration.
    - initial_learning_rate (float): Initial learning rate for the SGD updates.
    - tolerance (float): Convergence threshold.
    - d (float): Damping factor.
    - full_batch_update_iters (int): Number of iterations for the full-batch update phase.
    - dip_window (int): Window size for smoothing L2 norms.
    - plateau_iterations (int): Number of consecutive iterations where the gradient should remain stable for early stopping.
    - sampling_method (str): Method to sample nodes ('probability_based' or 'cyclic').

    Returns:
    - numpy.ndarray: The PageRank score for each node in the graph.
    - dict: L2 norms for each iteration.
    """
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

    # Initialize the size of the matrix
    N = M.shape[1]

    # Initialize PageRank vector with random values and normalize
    if init_vect is None:
        print("No pre-rank vector provided, proceeding with randomised intialisation")
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
    else:
        print("Pre-initialised vector provided")
        v = init_vect

    # Initialize last PageRank vector to infinity for convergence checks
    last_v = np.ones((N, 1)) * np.inf

    # Dictionary to store L2 norms for each iteration
    l2_dic = {}

    # Set to keep track of visited nodes (for cyclic sampling)
    visited_nodes = set()

    # Initialize counters and lists for plateau and dip detection
    plateau_count = 0
    dips_detected = 0
    dip_positions = []

    # Initialize an array to keep track of node visit counts (for probability-based sampling)
    visited_counts = np.zeros(N)

    for iter_ in range(num_iterations):
        # Decay the learning rate to ensure convergence
        learning_rate = initial_learning_rate / ((1 + iter_) / 10)

        # Probability-based sampling
        if sampling_method == "probability_based":
            probabilities = 1 / (1 + visited_counts)
            probabilities /= probabilities.sum()
            mini_batch_indices = np.random.choice(
                N, size=mini_batch_size, replace=False, p=probabilities
            )

        # Cyclic sampling
        elif sampling_method == "cyclic":
            if len(visited_nodes) < N:
                remaining_nodes = list(set(range(N)) - visited_nodes)
                mini_batch_indices = np.random.choice(
                    remaining_nodes,
                    size=min(mini_batch_size, len(remaining_nodes)),
                    replace=False,
                )
            else:
                mini_batch_indices = np.random.choice(
                    N, size=mini_batch_size, replace=False
                )

            # Update the set of visited nodes
            visited_nodes.update(mini_batch_indices)

        # Update node visit counts
        visited_counts[mini_batch_indices] += 1

        # Extract the mini-batch from the matrix and the PageRank vector
        M_mini_batch = M[mini_batch_indices, :]
        v_mini_batch = v[mini_batch_indices]

        # Store the current PageRank values for convergence checks
        last_v = v_mini_batch

        # Update the PageRank values using the mini-batch
        v_mini_batch = d * (learning_rate * M_mini_batch @ v) + ((1 - d) / N)
        v[mini_batch_indices] = v_mini_batch

        # Compute and store the L2 norm of the difference between the current and last PageRank values
        l2_norm = np.linalg.norm(v_mini_batch - last_v)
        l2_dic[iter_] = l2_norm

        # Compute smoothed L2 norms for dip detection
        if iter_ > dip_window:
            smoothed_values = np.convolve(
                list(l2_dic.values()), np.ones(dip_window) / dip_window, mode="valid"
            )
            gradient = smoothed_values[-1] - smoothed_values[-2]

            # Detect dips in the smoothed L2 norms
            if gradient < -1.5 * np.std(smoothed_values):
                dips_detected += 1
                dip_positions.append(iter_)

        # Check for convergence
        if l2_norm < tolerance:
            print("Converged at iteration {}".format(iter_))
            break

        # Early stopping based on smoothed L2 norms
        gradient_variance_window = 10
        if iter_ > gradient_variance_window:
            gradient_values = np.diff(smoothed_values)
            variance_of_gradient = np.var(gradient_values[-gradient_variance_window:])

            if sampling_method == "probability_based" and dips_detected == 1:
                if abs(gradient_values[-1]) < 0.3 * variance_of_gradient:
                    plateau_count += 1
                else:
                    plateau_count = 0

            elif sampling_method == "cyclic" and dips_detected > 1:
                if abs(gradient) < 0.5 * variance_of_gradient:
                    plateau_count += 1
                else:
                    plateau_count = 0

            # If the gradient has been stable for a number of iterations, stop early
            if plateau_count >= plateau_iterations:
                print(
                    f"Early stopping at iteration {iter_} due to plateau in L2 norm changes."
                )
                break

    # If the algorithm hasn't converged in the given number of iterations, display a message
    if iter_ == num_iterations - 1:
        print("pagerank model did not converge during the mini-batch phase")

    # Refine the PageRank values using full-batch updates
    print("Proceeding on to perform fine-tuning across full-batch")
    for _ in range(full_batch_update_iters):
        last_v_global = v.copy()
        v = d * (M @ v) + ((1 - d) / N)
        l2_norm_global = np.linalg.norm(v - last_v_global)
        l2_dic[iter_ + _ + 1] = l2_norm_global

    # Plot the L2 norms, smoothed L2 norms, dips, and detected plateaus
    plt.figure(figsize=(10, 6))
    plt.plot(list(l2_dic.keys()), list(l2_dic.values()), label="Original L2 Norm")
    smoothed_l2 = np.convolve(
        list(l2_dic.values()), np.ones(dip_window) / dip_window, mode="valid"
    )
    plt.plot(
        range(dip_window - 1, dip_window - 1 + len(smoothed_l2)),
        smoothed_l2,
        "r-",
        label="Smoothed L2 Norm",
    )

    for dip in dip_positions:
        plt.axvline(x=dip, color="g", linestyle="--")

    if plateau_count >= plateau_iterations:
        plt.axvspan(
            iter_ - plateau_count + 1,
            iter_,
            color="yellow",
            alpha=0.2,
            label="Detected Plateau",
        )

    # Highlight the global fine-tuning iterations
    plt.axvspan(
        iter_ + 1,
        iter_ + full_batch_update_iters + 1,
        color="blue",
        alpha=0.1,
        label="Global Fine-Tuning Iterations",
    )

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm")
    plt.title("Convergence of PageRank")
    plt.legend()
    plt.grid(True)
    plt.show()

    if sampling_method == "probability_based":
        print(
            "You should observe one dip in the graph, the first post initialisation and a continual trend downwards as the model learns more structure in your data"
        )
        print(
            "Erratic behavious post this initial dip should trend downwards. This shows that as the model visits more nodes, we see gradual model improvement"
        )
    elif sampling_method == "cyclic":
        print(
            "You should observe two dips in the graph, the first post initialisation and the second when the model starts to learn some structure and making informed updates"
        )
        print(
            "Erratic behavious post this second dip should trend downwards. This shows that dispite having visited all nodes and thus oscillating, we still see gradual model improvement"
        )

    return v, l2_dic


def SGDpagerank(
    M,
    num_iterations=1000,
    mini_batch_size=1000,
    initial_learning_rate=0.85,
    ignore_rate_iter=None,
    tolerance=1e-5,
    d=0.85,
    full_batch_update_iters=100,
    dip_window=5,
    plateau_iterations=5,
    sampling_method="probability_based",
    init_vect=None,
    smooth_updates=False,
    **kwargs,
):
    """
    Calculate the PageRank of each node in a graph using a mini-batch SGD approach.

    Parameters:
    - M (scipy.sparse.csr_matrix): The adjacency matrix of the graph.
    - num_iterations (int): The maximum number of iterations to perform.
    - mini_batch_size (int): Number of nodes to sample in each iteration.
    - initial_learning_rate (float): Initial learning rate for the SGD updates.
    - tolerance (float): Convergence threshold.
    - d (float): Damping factor.
    - full_batch_update_iters (int): Number of iterations for the full-batch update phase.
    - dip_window (int): Window size for smoothing L2 norms.
    - plateau_iterations (int): Number of consecutive iterations where the gradient should remain stable for early stopping.
    - sampling_method (str): Method to sample nodes ('probability_based' or 'cyclic').
    - smooth_updates (bool): Method to apply updates, incrementally or direct mini_batch updates.

    Returns:
    - numpy.ndarray: The PageRank score for each node in the graph.
    - dict: L2 norms for each iteration.

    """
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

    # Initialize the size of the matrix
    N = M.shape[1]

    # Initialize PageRank vector with random values and normalize
    if init_vect is None:
        print("No pre-rank vector provided, proceeding with randomised intialisation")
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
    else:
        print("Pre-initialised vector provided")
        v = init_vect

    # Initialize last PageRank vector to infinity for convergence checks
    last_v = np.ones((N, 1)) * np.inf

    # Dictionary to store L2 norms for each iteration
    l2_dic = {}

    # Set to keep track of visited nodes (for cyclic sampling)
    visited_nodes = set()

    # Initialize counters and lists for plateau and dip detection
    plateau_count = 0
    dips_detected = 0
    dip_positions = []

    # Initialize an array to keep track of node visit counts (for probability-based sampling)
    visited_counts = np.zeros(N)

    # Set ignore_rate_iter to num_iterations/10 if it's None, otherwise use the provided value
    ignore_rate_iter = (
        ignore_rate_iter if ignore_rate_iter is not None else num_iterations // 10
    )
    print("ignore_rate_iter has defaulted to {}".format(ignore_rate_iter))

    for iter_ in range(num_iterations):
        # add a parameter to ignore learn rate for first n iters
        if iter_ < ignore_rate_iter:
            learning_rate = 1
        elif iter_ >= ignore_rate_iter:
            # Decay the learning rate to ensure convergence
            learning_rate = initial_learning_rate / (
                (1 + (iter_ - ignore_rate_iter)) / 100
            )

        # Probability-based sampling
        if sampling_method == "probability_based":
            probabilities = 1 / (1 + visited_counts)
            probabilities /= probabilities.sum()
            mini_batch_indices = np.random.choice(
                N, size=mini_batch_size, replace=False, p=probabilities
            )

        # Cyclic sampling
        elif sampling_method == "cyclic":
            if len(visited_nodes) < N:
                remaining_nodes = list(set(range(N)) - visited_nodes)
                mini_batch_indices = np.random.choice(
                    remaining_nodes,
                    size=min(mini_batch_size, len(remaining_nodes)),
                    replace=False,
                )
            else:
                mini_batch_indices = np.random.choice(
                    N, size=mini_batch_size, replace=False
                )

            # Update the set of visited nodes
            visited_nodes.update(mini_batch_indices)

        # Update node visit counts
        visited_counts[mini_batch_indices] += 1

        # Extract the mini-batch from the matrix and the PageRank vector
        M_mini_batch = M[mini_batch_indices, :]
        # v_mini_batch = v[mini_batch_indices]

        # Store the current PageRank values for convergence checks
        last_v = v[mini_batch_indices].copy()

        # Update the PageRank values using the mini-batch
        v_mini_batch = d * (learning_rate * M_mini_batch @ v)

        if smooth_updates == True:
            v[mini_batch_indices] += v_mini_batch
        else:
            v[mini_batch_indices] = v_mini_batch

        # Normalize the full PageRank vector
        v += (1 - d) / N  # add the teleportaion probability
        v = v / np.linalg.norm(v, 1)

        # Compute and store the L2 norm of the difference between the current and last PageRank values
        l2_norm = np.linalg.norm(v[mini_batch_indices] - last_v)
        l2_dic[iter_] = l2_norm

        # Compute smoothed L2 norms for dip detection
        if iter_ > dip_window:
            smoothed_values = np.convolve(
                list(l2_dic.values()), np.ones(dip_window) / dip_window, mode="valid"
            )
            gradient = smoothed_values[-1] - smoothed_values[-2]

            # Detect dips in the smoothed L2 norms
            if gradient < -1.5 * np.std(smoothed_values):
                dips_detected += 1
                dip_positions.append(iter_)

        # Check for convergence
        if l2_norm < tolerance:
            print("Converged at iteration {}".format(iter_))
            break

        # Early stopping based on smoothed L2 norms
        gradient_variance_window = 10
        if iter_ > gradient_variance_window:
            gradient_values = np.diff(smoothed_values)
            variance_of_gradient = np.var(gradient_values[-gradient_variance_window:])

            if sampling_method == "probability_based" and dips_detected == 1:
                if abs(gradient_values[-1]) < 0.3 * variance_of_gradient:
                    plateau_count += 1
                else:
                    plateau_count = 0

            elif sampling_method == "cyclic" and dips_detected > 1:
                if abs(gradient) < 0.5 * variance_of_gradient:
                    plateau_count += 1
                else:
                    plateau_count = 0

            # If the gradient has been stable for a number of iterations, stop early
            if plateau_count >= plateau_iterations:
                print(
                    f"Early stopping at iteration {iter_} due to plateau in L2 norm changes."
                )
                break

    # If the algorithm hasn't converged in the given number of iterations, display a message
    if iter_ == num_iterations - 1:
        print("pagerank model did not converge during the mini-batch phase")

    # Refine the PageRank values using full-batch updates
    print("Proceeding on to perform fine-tuning across full-batch")
    for _ in range(full_batch_update_iters):
        last_v_global = v.copy()
        v = d * (M @ v) + ((1 - d) / N)
        l2_norm_global = np.linalg.norm(v - last_v_global)
        l2_dic[iter_ + _ + 1] = l2_norm_global

    # Plot the L2 norms, smoothed L2 norms, dips, and detected plateaus
    plt.figure(figsize=(10, 6))
    plt.plot(list(l2_dic.keys()), list(l2_dic.values()), label="Original L2 Norm")
    smoothed_l2 = np.convolve(
        list(l2_dic.values()), np.ones(dip_window) / dip_window, mode="valid"
    )
    plt.plot(
        range(dip_window - 1, dip_window - 1 + len(smoothed_l2)),
        smoothed_l2,
        "r-",
        label="Smoothed L2 Norm",
    )

    for dip in dip_positions:
        plt.axvline(x=dip, color="g", linestyle="--")

    if plateau_count >= plateau_iterations:
        plt.axvspan(
            iter_ - plateau_count + 1,
            iter_,
            color="yellow",
            alpha=0.2,
            label="Detected Plateau",
        )

    # Highlight the global fine-tuning iterations
    plt.axvspan(
        iter_ + 1,
        iter_ + full_batch_update_iters + 1,
        color="blue",
        alpha=0.1,
        label="Global Fine-Tuning Iterations",
    )

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm")
    plt.title("Convergence of PageRank")
    plt.legend()
    plt.grid(True)
    plt.show()

    if sampling_method == "probability_based":
        print(
            "You should observe one dip in the graph, the first post initialisation and a continual trend downwards as the model learns more structure in your data"
        )
        print(
            "Erratic behavious post this initial dip should trend downwards. This shows that as the model visits more nodes, we see gradual model improvement"
        )
    elif sampling_method == "cyclic":
        print(
            "You should observe two dips in the graph, the first post initialisation and the second when the model starts to learn some structure and making informed updates"
        )
        print(
            "Erratic behavious post this second dip should trend downwards. This shows that dispite having visited all nodes and thus oscillating, we still see gradual model improvement"
        )

    return v, l2_dic


def empirical_bayes_balanced_stratified_KNN_sampling(
    adata,
    feat_use,
    knn_key,
    sampling_rate=0.1,
    iterations=1,
    representation_priority=0.9,
    equal_allocation=False,
    replace=True,
    weight_penalty="laplacian",
    pl_label_entropy=False,
    resample_clones=False,
    n_hops=2,
    **kwargs,
):
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())
    if equal_allocation:
        print(
            "You are using an equal allocation mode of sampling, be warned that this can cause errors if the smaller populations are insufficient in number, consider replace == True"
        )

    if replace == True:
        print(
            "You are using sampling with replacement, this allows the model to create clones of cells"
        )

    if representation_priority > 0.8:
        print(
            "warning: you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )
        warnings.warn(
            "warning you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )

    # Convert string labels to integer labels
    unique_labels, indices = np.unique(adata.obs[feat_use], return_inverse=True)
    adata.obs["int.labels"] = indices

    # Calculate frequencies (prior probabilities)
    label_counts = np.bincount(indices)
    frequencies = label_counts / label_counts.sum()

    # Create a dictionary where keys are label indices and values are their frequencies (prior probabilities)
    prior_distribution = dict(zip(range(len(unique_labels)), frequencies))

    neighborhood_matrix = adata.obsp[adata.uns[knn_key]["connectivities_key"]]

    # Experimental feature with 1 hop matrix
    neighborhood_matrix = (
        neighborhood_matrix**n_hops
    )  ## N hops shopuld be added as an option in next iter. this controls the coverage of the graph

    # Initialize label probabilities with prior distribution
    label_probs = prior_distribution.copy()

    # Get indices for each label
    label_indices = {
        label: np.where(adata.obs["int.labels"] == label)[0]
        for label in range(len(unique_labels))
    }

    # Calculate total sample size and sample size per label for equal allocation
    total_sample_size = int(sampling_rate * adata.shape[0])
    sample_size_per_label = total_sample_size // len(unique_labels)

    if weight_penalty == "entropy_distance":
        print(
            "Using distance-entropy penalisation weights, this module is multi-threaded and quite compute intensive. If facing issues, use connectivity_ratio instead"
        )
        # Calculate entropy for each neighborhood in advance
        all_labels = adata.obs["int.labels"].values
        neighborhood_indices = [
            np.nonzero(neighborhood_matrix[idx])[1] for idx in range(adata.shape[0])
        ]
        # Calculate entropy for each cell in parallel
        import multiprocessing

        with multiprocessing.Pool() as pool:
            neighborhood_entropies = np.array(
                list(
                    pool.map(
                        calculate_entropy,
                        [(all_labels, idx) for idx in neighborhood_indices],
                    )
                )
            )

    elif weight_penalty == "laplacian":  # This is essentially an attention score
        print(
            "Using Laplacian penalty term, this is similar in concept to an attention score in GANs"
        )
        # This is essentially the calculation of the Laplacian of the graph.
        # Calculate degrees
        degrees = (
            np.array(neighborhood_matrix.sum(axis=1)).flatten() + 1
        )  # this is a generalization of the concept of degree for weighted graphs
        # Calculate inverse square root of degrees
        inv_sqrt_degrees = 1 / np.sqrt(degrees)
        # Create diagonal matrix of inverse square root degrees
        inv_sqrt_deg_matrix = scipy.sparse.diags(inv_sqrt_degrees)
        # Apply transformation to the adjacency matrix
        normalized_matrix = (
            inv_sqrt_deg_matrix @ neighborhood_matrix @ inv_sqrt_deg_matrix
        )

    elif (
        weight_penalty == "laplacian_SGD_pagerank"
    ):  # This is essentially an attention score
        print(
            "Using Laplacian-SGD-Pagerank penalty term, this is similar in concept to an attention score in GANs but incorperates stochastic gradient descent version of pagerank"
        )
        # This is essentially the calculation of the Laplacian of the graph.
        # Calculate degrees
        degrees = (
            np.array(neighborhood_matrix.sum(axis=1)).flatten() + 1
        )  # this is a generalization of the concept of degree for weighted graphs
        # Calculate inverse square root of degrees
        inv_sqrt_degrees = 1 / np.sqrt(degrees)
        # Create diagonal matrix of inverse square root degrees
        inv_sqrt_deg_matrix = scipy.sparse.diags(inv_sqrt_degrees)
        # Apply transformation to the adjacency matrix
        normalized_matrix = (
            inv_sqrt_deg_matrix @ neighborhood_matrix @ inv_sqrt_deg_matrix
        )
        #     # Now you can use normalized_matrix in place of neighborhood_matrix
        #     attention_score = normalized_matrix[indices].sum(axis = 1)
        # Convert your sparse matrix to a csr_matrix if it's not already
        csr_matrix = normalized_matrix.tocsr()
        attention_scores, l2_norm_dic = SGDpagerank(
            csr_matrix, **kwargs
        )  # num_iterations=1000,sampling_method='probability_based', mini_batch_size=1000, initial_learning_rate=0.85, tolerance=1e-6, d=0.85, full_batch_update_iters=100,

    # Create a dictionary to store the neighborhood entropy for each label at each iteration
    neighborhood_entropies_iter = {label: [] for label in range(len(unique_labels))}
    sampling_probabilities_over_iterations = np.zeros((iterations, len(unique_labels)))
    for _ in range(iterations):
        print("Iteration: {}".format(_))
        # Stratified sampling within each neighborhood for each label
        all_weights = []
        all_indices = []
        for label in label_indices.keys():
            indices = label_indices[label]
            other_indices = [
                idx
                for key, indices in label_indices.items()
                if key != label
                for idx in indices
            ]

            neighborhoods = neighborhood_matrix[
                indices
            ]  # select neighborhoods for the current label

            # Here we determine the liklighood that a sampled cell forms consistant neigthborhood
            # We take the sum of weights of the same labels within each neighborhood to represent the liklihood of a cell forming tight communities (px)
            # We divide p by the sum of weights of different labels in the same neighborhood to represent the liklihood of there being more than one state sharing the neighborhood. (nx)
            # We take the ratio of (px/nx) to represent the liklihood that a label represents consistant neighborhoods that are sufficiently independent
            # e.g The sum of weights of cells with different labels in a neighborhood relative to our observed cell should be low if the label for the cell is good. thus the ratio should be high.
            if weight_penalty == "connectivity_ratio":
                same_label_mask = np.array(
                    adata.obs["int.labels"][indices] == label, dtype=int
                )  # get mask for same-label cells
                same_label_mask = scipy.sparse.diags(
                    same_label_mask
                )  # convert to diagonal matrix for multiplication
                same_label_neighborhoods = neighborhoods[
                    :, indices
                ]  # get neighborhoods of same-label cells
                different_label_neighborhoods = neighborhoods[
                    :, other_indices
                ]  # get neighborhoods of different-label cells
                same_label_weights = np.array(
                    same_label_neighborhoods.sum(axis=1)
                ).ravel()
                different_label_weights = np.array(
                    different_label_neighborhoods.sum(axis=1)
                ).ravel()
                # we now compute a hybrid weighting scheme, where cells with more defined neigthborhood structure are prioritised. Howevever we do not want this over penalise, so we recover underrepresented cells with a inverse weighting parameter
                # Calculate the ratio of same-label weights to different-label weights
                # Add a small constant in the denominator to avoid division by zero
                weights = (same_label_weights) / (
                    different_label_weights + 1e-8
                )  # if same label sum distances are 0, do not consider this cell
                weights = weights / np.sum(
                    weights
                )  # this normnalisation means that every label has normalised set of weights to bias selection

            # Here we determine the liklighood that a sampled cell forms consistant neigthborhood using the mean distance to all neighbors * by the entropy of the neighborhood

            if weight_penalty == "entropy_distance":
                # We take neighborhoods that share the same labels and compute the sum of weights between same labels
                #             np.array(neighborhoods[:, indices].sum(axis=1)).ravel()
                same_label_mask = np.array(
                    adata.obs["int.labels"][indices] == label, dtype=int
                )  # get mask for same-label cells
                same_label_mask = scipy.sparse.diags(
                    same_label_mask
                )  # convert to diagonal matrix for multiplication
                same_label_neighborhoods = neighborhoods[:, indices]
                weights = (
                    np.array(same_label_neighborhoods.sum(axis=1)).ravel() + 1e-8
                )  # We take the sum of weights to all neighbors of the same label here
                # 1/weights give us the inverse where big weights are big distances
                weights *= 1 / (
                    neighborhood_entropies[indices] + 1e-8
                )  # use pre-computed entropies
                #             weights = weights
                weights = weights / np.sum(
                    weights
                )  # this normnalisation means that every label has normalised set of weights to bias selection

            elif (
                weight_penalty == "laplacian"
            ):  # This is essentially an attention score
                # This is essentially the calculation of the Laplacian of the graph.
                # Compute the attention or importance of each cell to their neighbors
                attention_scores = normalized_matrix[indices].sum(axis=1)
                weights = (np.array(attention_scores)).flatten()
            #                 weights = weights*(1*10**6)
            elif (
                weight_penalty == "laplacian_SGD_pagerank"
            ):  # This is essentially an attention score with pagerank and stochastic gradient descent
                # This is essentially the calculation of the Laplacian of the graph.
                # Compute the attention or importance of each cell to their neighbors
                attention_scores_ = [attention_scores[i] for i in indices]
                weights = (np.array(attention_scores_)).flatten()
            #                 weights = weights*(1*10**6)
            # Update weights based on representation priority and label probabilities
            # This should be a combination of the neighborhood-based weights and the label probability-based weights
            # we intriduce a iteration decay for each iteration for label probs here
            if representation_priority != 0:
                # weights = weights * ((((1 / (label_probs[label] + 1e-8))) **  (representation_priority)) )
                weights = weights * (
                    (
                        ((1 / (label_probs[label] + 1e-8)))
                        ** (representation_priority / (1 + _))
                    )
                )  # we added a decaying upsampling factor to prevent iterations from over sampling under re-presented states
            else:
                # weights = weights * (((1 / (label_probs[label] + 1e-8))))
                weights = weights * (((1 / (label_probs[label] + 1e-8))))
            # weights = np.array(weights) / np.sum(weights)  # normalization to probabilities
            all_weights.extend(weights)
            all_indices.extend(indices)
        all_weights = np.array(all_weights) / np.sum(
            all_weights
        )  # normalization to probabilities

        if resample_clones == True:
            sample_indices_n_dic = {}
            for _niter in range(0, 50):
                sample_indices_n = np.random.choice(
                    all_indices, size=total_sample_size, replace=replace, p=all_weights
                )
                sample_indices_n_dic[_niter] = sample_indices_n
            sample_indices_n_dic
            # Combine all the samples into one list
            combined_samples = np.hstack(list(sample_indices_n_dic.values()))
            # Count the frequency of each index in the combined samples
            index_counts = Counter(combined_samples)
            # Create a new weight array where the weight of each index is its original weight divided by its count
            new_weights = np.array(
                [
                    all_weights[i] / (1.0 + index_counts.get(index, 0))
                    for i, index in enumerate(all_indices)
                ]
            )
            # Normalize the new weights so they sum to 1
            new_weights /= new_weights.sum()
            # Sample from the distribution with the adjusted weights
            sample_indices = np.random.choice(
                all_indices, size=total_sample_size, replace=True, p=new_weights
            )
            # The result is a new sample where the frequently appearing indices in the initial samples have a lower chance of appearing
        else:
            sample_indices = np.random.choice(
                all_indices, size=total_sample_size, replace=replace, p=all_weights
            )

        # Update label probabilities based on the observed sample
        sample_labels = adata.obs["int.labels"][sample_indices]
        label_counts = np.bincount(sample_labels, minlength=len(unique_labels))
        label_probs = dict(
            zip(range(len(unique_labels)), label_counts / label_counts.sum() + 1e-8)
        )
        # Store the sampling probabilities for this iteration
        sampling_probabilities_over_iterations[_, :] = np.array(
            list(label_probs.values())
        )

        if pl_label_entropy == True:
            # Calculate the entropy for the sampled cells
            for label in label_indices.keys():
                # Get the indices of the sampled cells for the current label
                sampled_indices = [
                    idx
                    for idx in sample_indices
                    if adata.obs["int.labels"][idx] == label
                ]
                if sampled_indices:
                    # Get neighborhoods of the sampled cells
                    same_label_neighborhoods = neighborhood_matrix[sampled_indices]
                    # Get the indices of the connected cells
                    connected_indices = same_label_neighborhoods.nonzero()[1]
                    # Get the labels of the connected cells
                    connected_labels = adata.obs["int.labels"].values[connected_indices]
                    # Calculate the entropy for the current label
                    label_counts = np.bincount(
                        connected_labels, minlength=len(unique_labels)
                    )
                    probabilities = label_counts / len(connected_indices)
                    entropy_val = entropy(probabilities)
                    neighborhood_entropies_iter[label].append(entropy_val)
                else:
                    neighborhood_entropies_iter[label].append(None)

    average_sampling_probabilities = sampling_probabilities_over_iterations.mean(axis=0)
    updated_label_probs = dict(
        zip(range(len(unique_labels)), average_sampling_probabilities)
    )

    # Final stratified sampling using the last label_probs
    label_sample_sizes = {
        label: int(updated_label_probs[label] * total_sample_size)
        for label in range(len(unique_labels))
    }
    # Adjust sample sizes so total equals 'total_sample_size'
    difference = total_sample_size - sum(label_sample_sizes.values())
    label_sample_sizes[0] += difference  # adjust the first label for simplicity
    final_sample_indices = []

    if equal_allocation == True:
        label_sample_sizes = {
            label: sample_size_per_label for label in range(len(unique_labels))
        }

    for label, sample_size in label_sample_sizes.items():
        indices = label_indices[label]
        neighborhoods = neighborhood_matrix[indices][
            :, indices
        ]  # select neighborhoods for the current label

        same_label_mask = np.array(
            adata.obs["int.labels"][indices] == label, dtype=int
        )  # get mask for same-label cells
        same_label_mask = scipy.sparse.diags(
            same_label_mask
        )  # convert to diagonal matrix for multiplication

        same_label_neighborhoods = (
            same_label_mask @ neighborhoods @ same_label_mask
        )  # get neighborhoods of same-label cells
        different_label_neighborhoods = (
            neighborhoods - same_label_neighborhoods
        )  # get neighborhoods of different-label cells

        same_label_weights = np.array(same_label_neighborhoods.sum(axis=1)).ravel()
        different_label_weights = np.array(
            different_label_neighborhoods.sum(axis=1)
        ).ravel()

        # Calculate the ratio of same-label weights to different-label weights
        # Add a small constant in the denominator to avoid division by zero
        # weights = same_label_weights / (different_label_weights + 1e-8)
        #     weights = weights / weights.sum()  # normalization to probabilities
        specific_weights = np.array(all_weights[indices]) / np.sum(all_weights[indices])

        if resample_clones == True:
            #         sample_indices = np.random.choice(indices, size=sample_size, replace=replace, p=specific_weights)
            #         print('prior Non-Clone proportion == {}'.format( (len(list(set(sample_indices)))/len(sample_indices))))
            sample_indices_n_dic = {}
            for _niter in range(0, 50):
                try:
                    sample_indices_n = np.random.choice(
                        indices, size=sample_size, replace=replace, p=specific_weights
                    )
                except:
                    print(
                        "Warning -- sampling for {} without replacement failed, defaulting to taking all cells in this category".format(
                            label
                        )
                    )
                    sample_indices_n = np.random.choice(
                        indices, size=len(indices), replace=replace, p=specific_weights
                    )
                sample_indices_n_dic[_niter] = sample_indices_n
            sample_indices_n_dic
            # Combine all the samples into one list
            combined_samples = np.hstack(list(sample_indices_n_dic.values()))
            # Count the frequency of each index in the combined samples
            index_counts = Counter(combined_samples)
            # Create a new weight array where the weight of each index is its original weight divided by its count
            new_weights = np.array(
                [
                    specific_weights[i] / (1.0 + index_counts.get(index, 0))
                    for i, index in enumerate(indices)
                ]
            )
            # Normalize the new weights so they sum to 1
            new_weights /= new_weights.sum()
            # Sample from the distribution with the adjusted weights
            sampled_indices = np.random.choice(
                indices, size=sample_size, replace=True, p=new_weights
            )
            # The result is a new sample where the frequently appearing indices in the initial samples have a lower chance of appearing
        #         print('resampled Non-Clone proportion == {}'.format( (len(list(set(sample_indices)))/len(sample_indices))))
        else:
            sampled_indices = np.random.choice(
                indices, size=sample_size, replace=replace, p=specific_weights
            )

        #     sampled_indices = np.random.choice(indices, size=sample_size, replace=replace, p=specific_weights)
        final_sample_indices.extend(sampled_indices)
    adata_samp = adata[final_sample_indices, :]
    all_weights

    if pl_label_entropy == True:
        # plot entropy change per iteration
        # Calculate the number of columns for the legend
        ncol = math.ceil(
            len(unique_labels) / 20
        )  # Adjust the denominator to control the number of legend entries per column

        # Create a figure and an axes object
        fig, ax = plt.subplots(
            figsize=(5 + ncol, 5)
        )  # Adjust as needed. The width of the axes object will be always 5.

        # Compute the initial entropies for the whole dataset
        initial_entropies = {}
        for label in label_indices.keys():
            indices = label_indices[label]
            same_label_neighborhoods = neighborhood_matrix[indices]
            connected_indices = same_label_neighborhoods.nonzero()[1]
            connected_labels = adata.obs["int.labels"].values[connected_indices]
            label_counts = np.bincount(connected_labels, minlength=len(unique_labels))
            probabilities = label_counts / len(connected_indices)
            entropy_val = entropy(probabilities)
            initial_entropies[label] = entropy_val

        # Plot the change in neighborhood entropy over iterations for each label
        for label, entropies in neighborhood_entropies_iter.items():
            # Prepend the initial entropy to the list of entropies
            all_entropies = [initial_entropies[label]] + entropies
            ax.plot(
                range(len(all_entropies)), all_entropies, label=unique_labels[label]
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Scaled Neighborhood Entropy")
        # Set the y-axis to logarithmic scale
        ax.set_yscale("log")
        # Place the legend outside the plot, scaled with the height of the plot and spread into columns
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0, ncol=ncol
        )
        plt.tight_layout()
        plt.show()

    weights_out = {}
    weights_out["all_weights"] = all_weights
    weights_out["all_indices"] = all_indices

    # temp patch to include sf_attention
    # adata_samp = adata_samp.to_memory() # copy samp into memory
    # adata.obs['sf_attention'] = all_weights
    # adata_samp.obs['sf_attention'] = 0
    # adata_samp.obs['sf_attention'] = adata.obs.loc[adata.obs.index.isin(adata_samp.obs.index),'sf_attention']

    adata.obs["sf_attention"] = all_weights
    weights_out["adata_samp_sf_attention"] = pd.DataFrame(index=adata_samp.obs.index)
    weights_out["adata_samp_sf_attention"]["sf_attention"] = "0"
    weights_out["adata_samp_sf_attention"]["sf_attention"] = adata.obs.loc[
        adata.obs.index.isin(weights_out["adata_samp_sf_attention"].index),
        "sf_attention",
    ]

    return adata_samp, final_sample_indices, weights_out


def Attention_based_KNN_sampling(
    adata,
    knn_key,
    sampling_rate=0.1,
    iterations=1,
    representation_priority=0.9,
    equal_allocation=False,
    replace=True,
    weight_penalty="laplacian_SGD_pagerank",
    pl_label_entropy=False,
    resample_clones=False,
    n_hops=2,
    **kwargs,
):
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

    print("Non-stratified sampling based on attention weights chosen")

    if representation_priority > 0.8:
        print(
            "warning: you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )
        warnings.warn(
            "warning you have set a very high prioritisation factor, this will heavily bias the sampling of under-represented states"
        )

    neighborhood_matrix = adata.obsp[adata.uns[knn_key]["connectivities_key"]]

    # Experimental feature with 1 hop matrix
    # neighborhood_matrix = neighborhood_matrix**n_hops ## N hops shopuld be added as an option in next iter. this controls the coverage of the graph

    # Calculate total sample size and sample size per label for equal allocation
    total_sample_size = int(sampling_rate * adata.shape[0])
    #     sample_size_per_label = total_sample_size // len(unique_labels)

    if (
        weight_penalty == "laplacian_SGD_pagerank"
    ):  # This is essentially an attention score
        print(
            "Using Laplacian-SGD-Pagerank penalty term, this is similar in concept to an attention score in GANs but incorperates stochastic gradient descent version of pagerank"
        )
        # This is essentially the calculation of the Laplacian of the graph.
        # Calculate degrees
        degrees = (
            np.array(neighborhood_matrix.sum(axis=1)).flatten() + 1
        )  # this is a generalization of the concept of degree for weighted graphs
        # Calculate inverse square root of degrees
        inv_sqrt_degrees = 1 / np.sqrt(degrees)
        # Create diagonal matrix of inverse square root degrees
        inv_sqrt_deg_matrix = scipy.sparse.diags(inv_sqrt_degrees)
        # Apply transformation to the adjacency matrix
        normalized_matrix = (
            inv_sqrt_deg_matrix @ neighborhood_matrix @ inv_sqrt_deg_matrix
        )
        #     # Now you can use normalized_matrix in place of neighborhood_matrix
        #     attention_score = normalized_matrix[indices].sum(axis = 1)
        # Convert your sparse matrix to a csr_matrix if it's not already
        csr_matrix = normalized_matrix.tocsr()
        attention_scores, l2_norm_dic = SGDpagerank(
            csr_matrix, **kwargs
        )  # num_iterations=1000,sampling_method='probability_based', mini_batch_size=1000, initial_learning_rate=0.85, tolerance=1e-6, d=0.85, full_batch_update_iters=100,

    print("proceeding to 2 stage sampling using attention scores as priors")
    v = attention_scores.copy()
    attention_scores = attention_scores * (1 * 10**6)
    alpha = 1  # Dev note setting to higher avoids some issues due to very small numbers
    attention_scores = attention_scores**alpha
    # Add the attention scores to the observation dataframe
    adata.obs["sf_attention"] = attention_scores

    # Iterate over each unique stratifying variable
    # for n in adata.obs[strat_var].unique():
    tmp_obs = adata.obs  # [adata.obs[strat_var] == n]
    indices = range(len(tmp_obs))
    attention_scores_tmp = tmp_obs["sf_attention"]

    # Apply softmax to attention scores
    e_v = np.exp(attention_scores_tmp - np.max(attention_scores_tmp))
    sf_attention = e_v / e_v.sum()

    # Calculate total sample size
    total_sample_size = int(sampling_rate * len(tmp_obs))

    # Initialize list to store all sampled indices
    all_sampled_indices = []

    # Conduct sampling N times (100 by default)
    N = 100
    for i in range(N):
        sampled_indices = np.random.choice(
            indices, size=total_sample_size, replace=False, p=sf_attention
        )
        all_sampled_indices.extend(sampled_indices)

    # Count the frequency of each index
    index_freq = Counter(all_sampled_indices)

    # Convert the frequencies to probabilities
    total_samples = sum(index_freq.values())
    sampling_probabilities = {
        index: freq / total_samples for index, freq in index_freq.items()
    }

    # Convert dictionary keys and values to lists for sampling
    sample_indices = list(sampling_probabilities.keys())
    sample_probs = list(sampling_probabilities.values())

    sampled_indices_from_output = np.random.choice(
        sample_indices, size=total_sample_size, p=sample_probs, replace=False
    )

    adata_samp = adata[sampled_indices_from_output]
    print("Sampling complete!")
    weights_out = {}
    weights_out["all_weights"] = attention_scores
    weights_out["v"] = v
    weights_out["all_indices"] = all_sampled_indices
    # adata_samp = adata_samp.to_memory() # copy samp into memory

    weights_out["adata_samp_sf_attention"] = pd.DataFrame(index=adata_samp.obs.index)
    weights_out["adata_samp_sf_attention"]["sf_attention"] = "0"
    weights_out["adata_samp_sf_attention"]["sf_attention"] = adata.obs.loc[
        adata.obs.index.isin(weights_out["adata_samp_sf_attention"].index),
        "sf_attention",
    ]

    return adata_samp, sampling_probabilities, weights_out
