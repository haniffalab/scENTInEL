import scanpy as sc
import pandas as pd
import numpy as np
import scentinel as scent
import pickle as pkl
import matplotlib.pyplot as plt
#¢ new modules
import psutil
import time
import threading
from scipy.interpolate import UnivariateSpline
from queue import Queue, Empty
import scipy
import scipy.sparse as sp

def extract_connectivity_matrices(adata, adata_sub_dict, knn_key):
    """
    Extract the large matrix and the smaller matrices from the provided adata objects.
    """
    # Extract the main connectivity matrix from the main adata object
    main_matrix =  adata.obsp[adata.uns[model_params['knn_key']]['connectivities_key']]
    
    # Extract the smaller matrices from the subsetted adata objects
    sub_matrices = {sub:  sub_adata.obsp[sub_adata.uns[model_params['knn_key']]['connectivities_key']] for sub, sub_adata in adata_sub_dict.items()}
    
#     sub_adata.obsp[sub_adata.uns[model_params['knn_key']]['connectivities_key']]
#     adata.obsp[adata.uns[model_params['knn_key']]['connectivities_key']]
    return main_matrix, sub_matrices


def compute_density(matrix):
    """
    Compute the local density for each cell in the provided matrix.
    Density is defined as the sum of connectivity weights in a row.
    """
    return matrix.sum(axis=1).A1

def geometric_mean_weighted_update(main_matrix, sub_matrix, sub_density):
    """
    Update the main matrix using the geometric mean weighted by the local density from the sub_matrix.
    """
    rows, cols = sub_matrix.nonzero()
    for i, j in zip(rows, cols):
        # Compute the geometric mean weighted by the local density
        sub_weight = sub_matrix[i, j] * sub_density[i]
        main_weight = main_matrix[i, j] if i < main_matrix.shape[0] and j < main_matrix.shape[1] else 0
        update_weight = np.sqrt(main_weight * sub_weight)
        main_matrix[i, j] = update_weight

    return main_matrix


def get_node_mapping(main_barcodes, sub_barcodes):
    """
    Determine the mapping of nodes from the sub-matrix to the main matrix.
    """
    return {i: main_barcodes.index(sub_barcodes[i]) for i in range(len(sub_barcodes))}

def accumulate_subgraph_weights_normalized(main_matrix, adata_sub_dict, main_barcodes):
    """
    Accumulate the weighted edge values from all subgraphs into an accumulator matrix using normalized addition.
    """
    # Initialize the accumulator with zeros in CSR format and a count matrix
    accumulator = sp.lil_matrix(main_matrix.shape, dtype=np.float32)
    counts = sp.lil_matrix(main_matrix.shape, dtype=np.int32)
    
    for sub_key, sub_adata in adata_sub_dict.items():
        print('Processing subgraph {}'.format(sub_key))
        
        # Extract barcodes for the subgraph
        sub_barcodes = list(sub_adata.obs.index)
        
        # Determine the node mapping from subgraph to main graph
        mapping = get_node_mapping(main_barcodes, sub_barcodes)
        
        # Extract matrix from the sub_adata
        matrix = sub_adata.obsp[sub_adata.uns[model_params['knn_key']]['connectivities_key']]
        
        # Compute the density for the smaller matrix
        sub_density = compute_density(matrix)
        
        # Apply sigmoid scaling to the density information
        sub_density = 1 / (1 + np.exp(-sub_density))
        
        # Create a diagonal matrix with densities to multiply with the subgraph matrix
        density_diag = sp.diags(sub_density)
        
        # Compute the weighted matrix
        weighted_matrix = matrix.dot(density_diag)
        
        # Map the weighted matrix values to the accumulator using advanced indexing
        print("mapping the weighted matrix to the accumulator for subgraph {}".format(sub_key))
        i_sub, j_sub = weighted_matrix.nonzero()
        i_main = [mapping[i] for i in i_sub]
        j_main = [mapping[j] for j in j_sub]
        
        # Convert to CSR for efficient operations
        accumulator = accumulator.tocsr()
        counts = counts.tocsr()

        # Accumulate using addition and maintain counts without looping
        accumulator[i_main, j_main] += weighted_matrix[i_sub, j_sub].A1
        counts[i_main, j_main] += 1

        # Identify non-zero entries in counts
        nonzero_rows, nonzero_cols = counts.nonzero()

        # Normalize accumulator values by their counts without looping
        accumulator[nonzero_rows, nonzero_cols] = (accumulator[nonzero_rows, nonzero_cols].A * (1.0 / counts[nonzero_rows, nonzero_cols].A1))


    return accumulator.tocsr()

def integrate_matrices_normalized(adata, main_matrix, adata_sub_dict):
    """
    Integrate the information from all subgraphs into the main matrix.
    """
    # Extract barcodes for the main graph
    main_barcodes = list(adata.obs.index)
    
    # Accumulate the weighted edge values from all subgraphs
    accumulator = accumulate_subgraph_weights_normalized(main_matrix, adata_sub_dict, main_barcodes)

    # Identify overlapping non-zero entries between main_matrix and accumulator
    overlapping_nonzeros = main_matrix.multiply(accumulator)
    overlapping_rows, overlapping_cols = overlapping_nonzeros.nonzero()

    # Update weights for overlapping entries using advanced indexing
    overlapping_values_main = main_matrix[overlapping_rows, overlapping_cols].A1  # Convert matrix to 1D array
    overlapping_values_acc = accumulator[overlapping_rows, overlapping_cols].A1    # Convert matrix to 1D array

    updated_weights = (overlapping_values_main + overlapping_values_acc) / 2  ## update with geom mean

    # Update the main matrix with the updated weights
    main_matrix[overlapping_rows, overlapping_cols] = updated_weights

    # Add edges that are in accumulator but not in the main graph
    non_overlapping = accumulator - overlapping_nonzeros
    non_overlapping_rows, non_overlapping_cols = non_overlapping.nonzero()
    main_matrix[non_overlapping_rows, non_overlapping_cols] = non_overlapping[non_overlapping_rows, non_overlapping_cols].A1 

    # Ensure the matrix is symmetric and remove any self-loops
    main_matrix = (main_matrix + main_matrix.T) / 2
    main_matrix.setdiag(0)
    
    return main_matrix