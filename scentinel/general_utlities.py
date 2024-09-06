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
import logging
import os
import threading
import tracemalloc

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import scipy
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sns.set_theme(color_codes=True)


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
        peak_cpu = 0
        while self.running:
            peak_cpu = 0
            #           time.sleep(3)
            #             logging.info('CPU % usage = '+''+ str(currentProcess.cpu_percent(interval=1)))
            #             cpu_pct.append(str(currentProcess.cpu_percent(interval=1)))
            cpu = currentProcess.cpu_percent()
            # track the peak utilization of the process
            if cpu > peak_cpu:
                peak_cpu = cpu
                peak_cpu_per_core = peak_cpu / psutil.cpu_count()
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
def freq_redist_68CI(pred_out, clusters_reassign):
    """
    General description.

    Parameters:

    Returns:

    """
    freq_redist = clusters_reassign
    if freq_redist != False:
        logging.info("Frequency redistribution commencing")
        cluster_prediction = "consensus_clus_prediction"
        lr_predicted_col = "predicted"
        #         pred_out[clusters_reassign] = adata.obs[clusters_reassign].astype(str)
        reassign_classes = list(pred_out[clusters_reassign].unique())
        lm = 1  # lambda value
        pred_out[cluster_prediction] = pred_out[clusters_reassign]
        for z in pred_out[clusters_reassign][
            pred_out[clusters_reassign].isin(reassign_classes)
        ].unique():
            df = pred_out
            df = df[(df[clusters_reassign].isin([z]))]
            df_count = pd.DataFrame(df[lr_predicted_col].value_counts())
            # Look for classificationds > 68CI
            if len(df_count) > 1:
                df_count_temp = df_count[
                    df_count[lr_predicted_col]
                    > int(int(df_count.mean()) + (df_count.std() * lm))
                ]
                if len(df_count_temp >= 1):
                    df_count = df_count_temp
            # logging.info(df_count)
            freq_arranged = df_count.index
            cat = freq_arranged[0]
            # Make the cluster assignment first
            pred_out[cluster_prediction] = pred_out[cluster_prediction].astype(str)
            pred_out.loc[pred_out[clusters_reassign] == z, [cluster_prediction]] = cat
            # Create assignments for any classification >68CI
            for cats in freq_arranged:
                # logging.info(cats)
                cats_assignment = cats  # .replace(data1,'') + '_clus_prediction'
                pred_out.loc[
                    (pred_out[clusters_reassign] == z)
                    & (pred_out[lr_predicted_col] == cats),
                    [cluster_prediction],
                ] = cats_assignment
        min_counts = pd.DataFrame((pred_out[cluster_prediction].value_counts()))
        reassign = list(min_counts.index[min_counts[cluster_prediction] <= 2])
        pred_out[cluster_prediction] = pred_out[cluster_prediction].str.replace(
            str("".join(reassign)),
            str(
                "".join(
                    pred_out.loc[
                        pred_out[clusters_reassign].isin(
                            list(
                                pred_out.loc[
                                    (pred_out[cluster_prediction].isin(reassign)),
                                    clusters_reassign,
                                ]
                            )
                        ),
                        lr_predicted_col,
                    ]
                    .value_counts()
                    .head(1)
                    .index.values
                )
            ),
        )
        return pred_out


# Module to produce report for projection accuracy metrics on tranductive runs
def report_f1(model, train_x, train_label):
    ## Report accuracy score
    # ...
    # Report Precision score
    predicted_labels = model.predict(train_x)
    unique_labels = np.unique(np.concatenate((train_label, predicted_labels)))
    metric = pd.DataFrame(
        classification_report(train_label, predicted_labels, digits=2, output_dict=True)
    ).T
    cm = confusion_matrix(train_label, predicted_labels, labels=unique_labels)
    df_cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    df_cm = (df_cm / df_cm.sum(axis=0)) * 100
    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1)  # for label size
    pal = sns.diverging_palette(240, 10, n=10)
    # Plot precision recall and recall
    num_rows = len(metric.index)
    scale_factor = num_rows * 0.1  # scale factor depends on the number of rows
    bbox_y = (
        -0.4 - num_rows * 0.05
    )  # vertical position of the bbox depends on the number of rows
    bbox_height = num_rows * 0.05  # height of the bbox depends on the number of rows

    table = plt.table(
        cellText=metric.values,
        colWidths=[1] * len(metric.columns),
        rowLabels=metric.index,
        colLabels=metric.columns,
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=[0.25, bbox_y, 0.5, bbox_height],
    )
    table.scale(1, scale_factor)  # scale the table
    table.set_fontsize(10)

    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=pal)  # font size
    logging.info(classification_report(train_label, predicted_labels, digits=2))


# Generate psuedocells
def compute_local_scaling_factors(data, neighborhoods_matrix):
    """Compute local scaling factors based on total counts (UMIs) for each neighborhood."""
    total_counts_per_cell = data.sum(axis=1)
    avg_counts_neighborhood = neighborhoods_matrix.dot(
        total_counts_per_cell
    ) / neighborhoods_matrix.sum(axis=1)
    local_factors = total_counts_per_cell / neighborhoods_matrix.T.dot(
        avg_counts_neighborhood
    )
    return local_factors.A1


def compute_global_scaling_factors(data):
    """Compute global scaling factors based on total counts (UMIs) for all cells."""
    avg_counts = data.sum(axis=1).mean()
    return (data.sum(axis=1) / avg_counts).A1


def aggregate_data_single_load(
    adata, adata_samp, connectivity_matrix, method="local", **kwargs
):
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

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
    orig_obs_counts = pd.DataFrame(index=adata.obs_names, columns=["n_counts"])
    orig_obs_counts["n_counts"] = expression_matrix.sum(axis=1).A1

    # Apply scaling factors to individual cell expression profiles
    if method == "local":
        factors = compute_local_scaling_factors(expression_matrix, neighborhoods_matrix)
    elif method == "global":
        factors = compute_global_scaling_factors(expression_matrix)
    else:
        factors = np.ones(expression_matrix.shape[0])

    normalized_data = expression_matrix.multiply(np.reciprocal(factors)[:, np.newaxis])

    # Aggregate the normalized data
    aggregated_data = neighborhoods_matrix.dot(normalized_data)

    obs = adata.obs.iloc[indices]
    pseudobulk_adata = sc.AnnData(aggregated_data, obs=obs, var=adata.var)

    # Store original data neighbourhood identity
    pseudobulk_adata.uns["orig_data_connectivity_information"] = anndata.AnnData(
        X=connectivity_matrix,  # adata.obsp[adata.uns[knn_key]['connectivities_key']],
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.obs_names),
    )
    # Store original counts per cell
    pseudobulk_adata.obs["orig_counts_per_cell"] = orig_obs_counts

    # Store connectivity binary assignment
    # pseudobulk_adata.uns['orig_data_connectivity_information'].uns['neighbourhood_identity'] = ((adata.obsp[adata.uns[knn_key]['connectivities_key']][[adata.obs_names.get_loc(x) for x in pseudobulk_adata.obs_names], :]) > 0).astype(int)

    return pseudobulk_adata


def aggregate_data_v0_1_0(
    adata, adata_samp, connectivity_matrix, method="local", chunk_size=100
):
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
    if not is_backed and len(adata) < 1000000:
        # Use the regular approach if not in backed mode
        logging.info("Data is small enough to proceed with direct dot products")
        return aggregate_data_single_load(
            adata, adata_samp, connectivity_matrix, method
        )
    if adata_samp.isbacked:
        adata_samp = adata_samp.to_memory()

    logging.info("Data is too large to process in a single view, processing in chunks ")
    # Determine the number of chunks to process
    n_samples = adata_samp.shape[0]
    n_chunks = (n_samples + chunk_size - 1) // chunk_size  # Ceiling division
    aggregated_data_dict = {}
    obs_dict = {}

    orig_obs_counts = pd.DataFrame(index=adata.obs_names, columns=["n_counts"])

    # Loop through chunks with a progress bar
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples)
        current_chunk = adata_samp[start_idx:end_idx]
        obs_dict[chunk_idx] = adata_samp.obs.iloc[start_idx:end_idx]

        # Get indices of cells in the current chunk
        # indices = adata.obs.index.isin(current_chunk.obs.index).nonzero()[0]
        indices = adata.obs.index.get_indexer(current_chunk.obs.index)
        # Extract the corresponding neighborhood matrix
        neighborhoods_matrix_chunk = connectivity_matrix[indices]
        # Identify unique neighbor indices for cells in this chunk
        neighbor_indices = np.unique(neighborhoods_matrix_chunk.nonzero()[1])

        # Adjust neighborhood matrix to only cover relevant neighbors
        neighborhoods_matrix_chunk = connectivity_matrix[indices, :][
            :, neighbor_indices
        ]

        # Extract the expression matrix for these neighbors
        expression_matrix_chunk = adata[neighbor_indices].to_memory().X

        # Store original counts in dataframe
        orig_obs_counts.loc[
            adata[neighbor_indices].obs.index, "n_counts"
        ] = expression_matrix_chunk.sum(axis=1).A1

        # Calculate scaling factors based on the specified method
        if method == "local":
            factors = compute_local_scaling_factors(
                expression_matrix_chunk, neighborhoods_matrix_chunk
            )
        elif method == "global":
            factors = compute_global_scaling_factors(expression_matrix_chunk)
        elif method == "sum":
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(
                expression_matrix_chunk
            )
        else:
            factors = np.ones(expression_matrix_chunk.shape[0])

        if method != "sum":
            # Normalize data using scaling factors
            normalized_data_chunk = expression_matrix_chunk.multiply(
                np.reciprocal(factors)[:, np.newaxis]
            )

            # Aggregate the normalized data using the neighborhood matrix
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(
                normalized_data_chunk
            )

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
    aggregated_data_combined = scipy.sparse.vstack(
        [aggregated_data_dict[idx] for idx in ordered_chunks]
    )
    aggregated_obs = pd.concat([obs_dict[idx] for idx in ordered_chunks], axis=0)
    # Return as AnnData object
    return sc.AnnData(aggregated_data_combined, obs=aggregated_obs, var=adata.var)


def aggregate_data(
    adata, adata_samp, connectivity_matrix, method="local", chunk_size=100, **kwargs
):
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
    # Unpack kwargs
    if kwargs:
        for key, value in kwargs.items():
            globals()[key] = value
        kwargs.update(locals())

    # Check if in backed mode
    is_backed = adata.isbacked
    if not is_backed and len(adata) < 1000000:
        # Use the regular approach if not in backed mode
        logging.info("Data is small enough to proceed with direct dot products")
        return aggregate_data_single_load(
            adata, adata_samp, connectivity_matrix, method, **kwargs
        )
    if adata_samp.isbacked:
        adata_samp = adata_samp.to_memory()

    logging.info(
        "Data is too large to process in a single view or in backed mode, processing in chunks "
    )
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
        # indices = adata.obs.index.isin(current_chunk.obs.index).nonzero()[0]
        indices = adata.obs.index.get_indexer(current_chunk.obs.index)
        # Extract the corresponding neighborhood matrix
        neighborhoods_matrix_chunk = connectivity_matrix[indices]
        # Identify unique neighbor indices for cells in this chunk
        neighbor_indices = np.unique(neighborhoods_matrix_chunk.nonzero()[1])

        # Adjust neighborhood matrix to only cover relevant neighbors
        neighborhoods_matrix_chunk = connectivity_matrix[indices, :][
            :, neighbor_indices
        ]

        # Extract the expression matrix for these neighbors
        expression_matrix_chunk = adata[neighbor_indices].to_memory().X

        # Calculate scaling factors based on the specified method
        if method == "local":
            factors = compute_local_scaling_factors(
                expression_matrix_chunk, neighborhoods_matrix_chunk
            )
        elif method == "global":
            factors = compute_global_scaling_factors(expression_matrix_chunk)
        elif method == "sum":
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(
                expression_matrix_chunk
            )
        else:
            factors = np.ones(expression_matrix_chunk.shape[0])

        if method != "sum":
            # Normalize data using scaling factors
            normalized_data_chunk = expression_matrix_chunk.multiply(
                np.reciprocal(factors)[:, np.newaxis]
            )

            # Aggregate the normalized data using the neighborhood matrix
            aggregated_data_chunk = neighborhoods_matrix_chunk.dot(
                normalized_data_chunk
            )

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
    aggregated_data_combined = scipy.sparse.vstack(
        [aggregated_data_dict[idx] for idx in ordered_chunks]
    )
    aggregated_obs = pd.concat([obs_dict[idx] for idx in ordered_chunks], axis=0)

    # Create aggregated AnnData object
    pseudobulk_adata = sc.AnnData(
        aggregated_data_combined, obs=aggregated_obs, var=adata.var
    )

    # Store original data neighbourhood identity
    pseudobulk_adata.uns["orig_data_connectivity_information"] = anndata.AnnData(
        X=connectivity_matrix,  # adata.obsp[adata.uns[knn_key]['connectivities_key']],
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.obs_names),
    )
    #     # Store original counts per cell
    #     pseudobulk_adata.obs['orig_counts_per_cell'] = orig_obs_counts

    # Store connectivity binary assignment
    # pseudobulk_adata.uns['orig_data_connectivity_information'].uns['neighbourhood_identity'] = ((adata.obsp["connectivities"][[adata.obs_names.get_loc(x) for x in pseudo_bulk_data.obs_names], :]) > 0).astype(int)

    return pseudobulk_adata


import os
import time
from queue import Empty, Queue

import psutil
from scipy.interpolate import UnivariateSpline


class ResourceMonitor:
    def __init__(self):
        self.control_queue = Queue()
        self.sys_memory_data = []
        self.vms_memory_data = []
        self.rss_memory_data = []
        self.cpu_data = []
        self.time_data = []
        self.thread = None

    def monitor_resources(self, interval=1):
        """Monitor system resources at intervals."""
        pid = os.getpid()
        current_process = psutil.Process(pid)
        while True:
            try:
                message = self.control_queue.get(timeout=interval)
                if message == "stop":
                    break
            except Empty:
                pass
            memory_usage = current_process.memory_info()
            self.sys_memory_data.append(psutil.virtual_memory().used / (1024**2))
            self.rss_memory_data.append(memory_usage.rss / (1024**2))
            self.vms_memory_data.append(memory_usage.rss / (1024**2))
            self.cpu_data.append(psutil.cpu_percent(interval=None))
            self.time_data.append(time.time())

    def start_monitoring(self, interval=1):
        """Start the monitoring in a separate thread."""
        self.thread = threading.Thread(target=self.monitor_resources, args=(interval,))
        self.thread.start()

    def stop_monitoring(self):
        """Stop the monitoring."""
        self.control_queue.put("stop")
        if self.thread:
            self.thread.join()


def plot_resources(monitor, mem_key="rss_memory_data", cpu_key="cpu_data"):
    monitor.memory_data = getattr(monitor, mem_key)
    monitor.cpu_data = getattr(monitor, cpu_key)

    """Plot the collected resource data."""
    plt.figure(figsize=(12, 10))

    # Memory Usage Over Time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        monitor.time_data, monitor.memory_data, "-o", color="blue", label="Memory Usage"
    )
    ax1.fill_between(monitor.time_data, 0, monitor.memory_data, color="blue", alpha=0.3)
    ax1.set_ylim(
        min(monitor.memory_data) - 0.1 * min(monitor.memory_data),
        max(monitor.memory_data) + 0.1 * max(monitor.memory_data),
    )

    # Add smoothed spline for memory data
    spl_memory = UnivariateSpline(monitor.time_data, monitor.memory_data, s=100)
    ax1.plot(
        monitor.time_data,
        spl_memory(monitor.time_data),
        "k-",
        linewidth=2,
        label="Smoothed Memory Usage",
    )

    ax1.set_title("{} Usage Over Time".format(mem_key))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Memory (MB)")
    ax1.legend()

    # CPU Usage Over Time
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(monitor.time_data, monitor.cpu_data, "-o", color="red", label="CPU Usage")
    ax2.fill_between(monitor.time_data, 0, monitor.cpu_data, color="red", alpha=0.3)
    ax2.set_ylim(
        min(monitor.cpu_data) - 5, max(monitor.cpu_data) + 5
    )  # giving a buffer of 5% for CPU
    # Add smoothed spline for CPU data
    spl_cpu = UnivariateSpline(monitor.time_data, monitor.cpu_data, s=10)
    ax2.plot(
        monitor.time_data,
        spl_cpu(monitor.time_data),
        "k-",
        linewidth=2,
        label="Smoothed CPU Usage",
    )

    ax2.set_title("{} Usage Over Time".format(cpu_key))
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("CPU (%)")
    ax2.legend()

    # Histogram for Memory Usage
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(monitor.memory_data, bins=30, color="blue", alpha=0.7)
    ax3.set_title("Histogram of {} Usage".format(mem_key))
    ax3.set_xlabel("Memory (MB)")
    ax3.set_ylabel("Frequency")

    # Histogram for CPU Usage
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(monitor.cpu_data, bins=30, color="red", alpha=0.7)
    ax4.set_title("Histogram of {} Usage".format(cpu_key))
    ax4.set_xlabel("CPU (%)")
    ax4.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def celltype_threshold_checker(
    adata: anndata.AnnData,
    adata_samp: anndata.AnnData,
    threshold_cell_number: int,
    **kwargs,
):
    feat_use = kwargs.get("feat_use", None)

    # Get the value counts for the specified column in both AnnData objects
    value_counts_adata = adata.obs[feat_use].value_counts().reset_index()
    value_counts_adata_samp = adata_samp.obs[feat_use].value_counts().reset_index()

    # Rename the columns to distinguish between the two datasets
    value_counts_adata.columns = [feat_use, "Count (adata)"]
    value_counts_adata_samp.columns = [feat_use, "Count (adata_samp)"]

    # Merge the two DataFrames on the column of interest (e.g., 'col')
    merged_value_counts = pd.merge(
        value_counts_adata, value_counts_adata_samp, on=feat_use, how="outer"
    )

    # Fill NaN values with 0
    merged_value_counts = merged_value_counts.fillna(0)

    # Calculate additional columns
    merged_value_counts["Relative Proportion (adata)"] = (
        merged_value_counts["Count (adata)"]
        / merged_value_counts["Count (adata)"].sum()
    )
    merged_value_counts["Relative Proportion (adata_samp)"] = (
        merged_value_counts["Count (adata_samp)"]
        / merged_value_counts["Count (adata_samp)"].sum()
    )
    merged_value_counts["Percentage of Original Data Captured"] = (
        merged_value_counts["Count (adata_samp)"]
        / merged_value_counts["Count (adata)"].sum()
    ) * 100
    merged_value_counts["Delta Proportion Change"] = (
        merged_value_counts["Relative Proportion (adata_samp)"]
        - merged_value_counts["Relative Proportion (adata)"]
    )

    # Filter rows based on the threshold_cell_number
    filtered_rows = merged_value_counts[
        merged_value_counts["Count (adata_samp)"] < threshold_cell_number
    ]

    # Sort the DataFrame by the number of cells captured in adata_samp
    sorted_table = filtered_rows.sort_values(by="Count (adata_samp)", ascending=False)

    # Return tables
    return merged_value_counts, sorted_table


def update_label_anchors_by_lower_threshold(
    adata: anndata.AnnData,
    adata_samp: anndata.AnnData,
    threshold_cell_number: int,
    **kwargs,
):
    feat_use = kwargs.get("feat_use", None)

    # Get labels below the minimum cells to recover in adata_samp
    recovery_labels = [
        val
        for val in adata.obs[feat_use].unique()
        if val not in adata_samp.obs[feat_use].values
        or adata_samp.obs[feat_use].value_counts().get(val, 0) < threshold_cell_number
    ]

    # Check if there are any labels to process
    if not recovery_labels:
        logging.info(
            "No labels below the minimum cell recovery threshold. Resulting adata_samp has not changed."
        )
        return adata_samp

    else:
        l = len(recovery_labels)
        logging.info(
            f"{l} labels below the minimum cell recovery threshold. Recovering these labels based off attention score to return up to {threshold_cell_number} cells per label."
        )

        # Initialize indexes list
        indexes = []

        # Iterate over labels
        for label in recovery_labels:
            df = adata.obs[adata.obs[feat_use].isin([label])]

            # only consider cells which are not already in adata_samp
            df = df[~df.index.isin(list(adata_samp.obs.index))]

            # Check if there are enough cells for the specified label
            if len(df) < threshold_cell_number:
                indexes.extend(
                    list(df.sort_values(by="sf_attention", ascending=False).index)
                )

            else:
                indexes.extend(
                    list(
                        df.sort_values(by="sf_attention", ascending=False)
                        .head(threshold_cell_number)
                        .index
                    )
                )

        indexes = list(adata_samp.obs.index) + indexes
        adata_samp = adata[adata.obs.index.isin(indexes)]
        gc.collect()
        logging.info("Updated indexes for adata_samp completed")
        return adata_samp
