import os
import torch
import numpy as np
from torch_geometric.data import Data
from SIMLR_PY.SIMLR import SIMLR_LARGE
from SIMLR_PY.SIMLR.helper import fast_pca
import matplotlib.pyplot as plt
from config import MODEL_NAME, N_SOURCE_NODES, N_TARGET_NODES


def calc_norm_distances(predicted_CBT, X_test_target):
    """
        Calculates the norm distance between predicted CBT and each test subject

        Parameters:
            predicted_CBT (np.array): Predicted CBT of shape [N_TARGET_NODES x N_TARGET_NODES]
            X_test_target (list):     List of all test subjects with shape [N_SUBJECTS x N_FEAT]

        Returns:
            list: List of all norm distances
    """
    norm_dists = []
    for test_target in X_test_target:
        norm_dists.append(np.abs(predicted_CBT - target_antivectorize(test_target)))

    return norm_dists


def print_fold_specific_results(fold, evaluation_results_fold_specific, mean_norm_distance_fold_specific):
    """
        Prints the test results for the current fold

        Parameters:
            fold (int): Fold number
            evaluation_results_fold_specific (obj): An object containing:
                                                    MAE: Mean absolute error between predicted and target CBT
                                                    MAE(PR): Mean absolute error of pagerank centrality between predicted and target CBT
                                                    MAE(EC): Mean absolute error of eigenvector centrality between predicted and target CBT
                                                    MAE(BC): Mean absolute error of betweenness centrality between predicted and target CBT
            mean_norm_distance_fold_specific(double): Mean norm distance for this fold 
    """
    print()
    print(f"### TEST RESULTS FOLD {fold} BETWEEN PREDICTED AND GROUND-TRUTH CBT ###")
    print("MAE: ", str(evaluation_results_fold_specific["MAE"].item()))
    print("MAE(Pagerank Centrality) ", str(evaluation_results_fold_specific["MAE(PR)"].item()))
    print("MAE(Eigenvector Centrality) ", str(evaluation_results_fold_specific["MAE(EC)"].item()))
    print("MAE(Betweenness Centrality): ", str(evaluation_results_fold_specific["MAE(BC)"].item()))
    print("Mean Norm Distance Between Each Test Subject and Predicted CBT: " + str(mean_norm_distance_fold_specific))
    print()


def print_final_results(mae, mae_pr, mae_ec, mae_bc, mean_norm_dists):
    """
        Prints the final test results by calculating mean and std of evaluation results in each fold 

        Parameters:
            mae (list): List of MAEs in each fold
            mae_pr (list): List of MAEs of pagerank centralities in each fold
            mae_ec (list): List of MAEs of eigenvector centralities in each fold
            mae_bc (list): List of MAEs of betweenness centralities in each fold
            mean_norm_dists (list): List of mean norm distances in each fold
    """
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)

    mae_pr_mean = np.mean(mae_pr)
    mae_pr_std = np.std(mae_pr)

    mae_ec_mean = np.mean(mae_ec)
    mae_ec_std = np.std(mae_ec)

    mae_bc_mean = np.mean(mae_bc)
    mae_bc_std = np.std(mae_bc)

    mean_norm_dists_mean = np.mean(mean_norm_dists)
    mean_norm_dists_std = np.std(mean_norm_dists)

    print("### MEAN TEST RESULTS ACROSS FOLDS BETWEEN PREDICTED AND GROUND-TRUTH CBT ###")
    print("MAE: ", str(mae_mean) + " +/- " + str(mae_std))
    print("MAE(Pagerank Centrality)", str(mae_pr_mean) + " +/- " + str(mae_pr_std))
    print("MAE(Eigenvector Centrality)", str(mae_ec_mean) + " +/- " + str(mae_ec_std))
    print("MAE(Betweenness Centrality)", str(mae_bc_mean) + " +/- " + str(mae_bc_std))
    print("Mean of Mean Norm Distances Between Each Test Subject and Predicted CBT:", str(mean_norm_dists_mean) + " +/- " + str(mean_norm_dists_std))


def save_fig(fig, title, filename):
    """
        Saves the given figure in "output/{MODEL_NAME}" folder

        Parameters:
            fig (matplotlib.figure): Matplotlib figure to be saved
            title (string): Title of the figure
            filename (string): Filename of the figure
    """
    output_path = f"output/{MODEL_NAME}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.figure()
    plt.matshow(fig)
    plt.colorbar()
    plt.title(title)
    plt.savefig(output_path + "/" + filename)


def cluster_with_SIMLR(data, dgn_caster, n_clusters):
    """
        Splits the given data into clusters using SIMLR and cast each subject to proper shape for DGN input.

        Parameters:
            data (np.array): Data to be clustered into similar manifolds
            dgn_caster (function): Either cast_source_to_DGN_input or cast_target_to_DGN_input function
            n_clusters (int): Number of clusters

        Returns:
            list: List with length of n_clusters. Each element is one cluster of the data.
    """
    simlr = SIMLR_LARGE(n_clusters, 15)

    F1 = simlr.fit(fast_pca(data, 6))[1]
    cluster_labels = simlr.fast_minibatch_kmeans(F1, n_clusters)

    X_clusters = []

    for cluster_label in range(n_clusters):
        X_clusters.append(dgn_caster(data[cluster_labels == cluster_label]))

    return X_clusters


def create_simulated_data(N_SUBJECTS, N_SOURCE_NODES, N_TARGET_NODES):
    """
        Create simulated data using normal random distribution
        Data is the vectorized representation of shape [N_SUBJECTS x N_ROIs x N_ROIs]
        So, shape of the data is [N_SUBJECTS x N_FEAT]
        where N_FEAT is the number of features for each subject and is eqaul to N_ROIs x (N_ROIs-1) / 2
        Also saves the created simulated data in "data/simulated/{MODEL_NAME}" folder

        Parameters:
            N_SUBJECTS (int): Number of subjects
            N_SOURCE_NODES (int): Number of region of interests (N_ROIs) in source domain
            N_TARGET_NODES (int): Number of region of interests (N_ROIs) in target domain

        Returns:
            np.array, np.array: Returns a tuple where the first element is created simulated source domain
                                and the second element is created simulated target domain

    """
    X_s = np.random.normal(
        0, 0.5, (N_SUBJECTS, (N_SOURCE_NODES * (N_SOURCE_NODES - 1) // 2)))
    X_t = np.random.normal(
        0, 0.5, (N_SUBJECTS, (N_TARGET_NODES * (N_TARGET_NODES - 1) // 2)))

    # Save simulated data
    data_path = f"data/simulated/{MODEL_NAME}/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(data_path + "source_data.npy", X_s)
    np.save(data_path + "target_data.npy", X_t)

    return X_s, X_t


def source_antivectorize(source_sample):
    """
        Converts given source vector into adjacency matrix

        Parameters:
            source_sample (np.array): A source subject column vector with N_SOURCE_NODES * (N_SOURCE_NODES - 1) / 2 elements

        Returns:
            np.array: Returns adjacency matrix with shape [N_SOURCE_NODES x N_SOURCE_NODES]
    """
    adj_matrix = np.zeros((N_SOURCE_NODES, N_SOURCE_NODES))
    triu_i = np.triu_indices(N_SOURCE_NODES, 1)

    adj_matrix[triu_i] = source_sample
    adj_matrix = adj_matrix + adj_matrix.T

    # Normalize data
    adj_matrix /= np.max(adj_matrix)

    diag_i = np.diag_indices(N_SOURCE_NODES)
    adj_matrix[diag_i] = 1

    return adj_matrix


def target_antivectorize(target_sample):
    """
        Converts given target vector into adjacency matrix

        Parameters:
            target_sample (np.array): A target subject column vector with N_TARGET_NODES * (N_TARGET_NODES - 1) / 2 elements

        Returns:
            np.array: Returns adjacency matrix with shape [N_TARGET_NODES x N_TARGET_NODES]
    """
    adj_matrix = np.zeros((N_TARGET_NODES, N_TARGET_NODES))
    tril_i = np.tril_indices(N_TARGET_NODES, -1)

    adj_matrix[tril_i] = target_sample
    adj_matrix = adj_matrix + adj_matrix.T
    diag_i = np.diag_indices(N_TARGET_NODES)
    adj_matrix[diag_i] = 1

    adj_matrix[adj_matrix < 0] = 0

    return adj_matrix


def source_to_graph(source_sample_adj_matrix):
    """
        Converts given source adjacency matrix into torch.geometric Data object
        Three attributes of Data objects are filled:
                x: Node features with shape [number_of_nodes, number_of_node_features]
                edge_attr: Edge features with shape [number_of_edges, number_of_edge_features]
                edge_index: Graph connectivities with shape [2, number_of_edges] (COO format)

        Parameters:
            source_sample_adj_matrix (np.array): Adjacency matrix for the source sample

        Returns:
            torch.geometric.data.Data object: Data object representing the source graph
    """
    row0 = np.repeat(np.arange(N_SOURCE_NODES), N_SOURCE_NODES)
    row1 = np.tile(np.arange(N_SOURCE_NODES), N_SOURCE_NODES)
    edge_index = np.array([row0, row1])

    edge_attr = []

    for j in range(edge_index.shape[1]):
        edge_attr.append(
            source_sample_adj_matrix[edge_index[:, j][0], edge_index[:, j][1]])

    x = torch.from_numpy(source_sample_adj_matrix).float().requires_grad_(True)
    edge_index_tensor = torch.from_numpy(edge_index)
    edge_attr_tensor = torch.reshape(
        torch.tensor(edge_attr), (N_SOURCE_NODES * N_SOURCE_NODES, 1)).float().requires_grad_(True)

    return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)


def target_to_graph(target_sample_adj_matrix):
    """
        Converts given target adjacency matrix into torch.geometric Data object
        Three attributes of Data objects are filled:
                x: Node features with shape [number_of_nodes, number_of_node_features]
                edge_attr: Edge features with shape [number_of_edges, number_of_edge_features]
                edge_index: Graph connectivities with shape [2, number_of_edges] (COO format)

        Parameters:
            target_sample_adj_matrix (np.array): Adjacency matrix for the target sample

        Returns:
            torch.geometric.data.Data object: Data object representing the target graph
    """
    row0 = np.repeat(np.arange(N_TARGET_NODES), N_TARGET_NODES)
    row1 = np.tile(np.arange(N_TARGET_NODES), N_TARGET_NODES)
    edge_index = np.array([row0, row1])

    edge_attr = []

    for j in range(edge_index.shape[1]):
        edge_attr.append(
            target_sample_adj_matrix[edge_index[:, j][0], edge_index[:, j][1]])

    x = torch.from_numpy(target_sample_adj_matrix).float().requires_grad_(True)
    edge_index_tensor = torch.from_numpy(edge_index).long()
    edge_attr_tensor = torch.reshape(
        torch.tensor(edge_attr), (N_TARGET_NODES, 1)).float().requires_grad_(True)

    return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)


def cast_source_dataset(source_dataset):
    """
        Applies the source_antivectorize and source_to_graph functions on the whole source dataset

        Parameters:
            source_dataset (list): Source dataset with shape [N_SUBJECTS, N_FEAT]

        Returns:
            list: List of Data objects
    """
    casted_source_dataset = []

    for source_sample in source_dataset:
        adj_matrix = source_antivectorize(source_sample)
        graph = source_to_graph(adj_matrix)
        casted_source_dataset.append(graph)

    return casted_source_dataset


def cast_target_dataset(target_dataset):
    """
        Applies the target_antivectorize and target_to_graph functions on the whole target dataset
        
        Parameters:
            target_dataset (list): Target dataset with shape [N_SUBJECTS, N_FEAT]

        Returns:
            list: List of Data objects
    """
    casted_target_dataset = []

    for target_sample in target_dataset:
        adj_matrix = target_antivectorize(target_sample)
        graph = target_to_graph(adj_matrix)
        casted_target_dataset.append(graph)

    return casted_target_dataset


def cast_source_to_DGN_input(source_data):
    """
        DGN model accepts the source data with shape (N_SUBJECTS, N_SOURCE_NODES, N_SOURCE_NODES, N_VIEWS)
        Our TIS-Net framework works on only single-view networks.
        So, N_VIEWS = 1
        This function adds another dimension (N_VIEWS = 1) for the proper shape for DGN input

        Parameters:
            source_data (list): Source dataset with shape [N_SUBJECTS, N_FEAT]

        Returns:
            np.array: Array with shape [N_SUBJECTS, N_SOURCE_NODES, N_SOURCE_NODES, 1]
    """
    source_data_as_adj_matrices = []

    for source_sample in source_data:
        adj_matrix = source_antivectorize(source_sample)
        source_data_as_adj_matrices.append(adj_matrix)

    return np.array(source_data_as_adj_matrices).reshape((len(source_data_as_adj_matrices), N_SOURCE_NODES, N_SOURCE_NODES, 1))


def cast_target_to_DGN_input(target_data):
    """
        DGN model accepts the target data with shape (N_SUBJECTS, N_TARGET_NODES, N_TARGET_NODES, N_VIEWS)
        Our TIS-Net framework works on only single-view networks.
        So, N_VIEWS = 1
        This function adds another dimension (N_VIEWS = 1) for the proper shape for DGN input

        Parameters:
            target_data (list): Target dataset with shape [N_SUBJECTS, N_FEAT]

        Returns:
            np.array: Array with shape [N_SUBJECTS, N_TARGET_NODES, N_TARGET_NODES, 1]
    """
    target_data_as_adj_matrices = []

    for target_sample in target_data:
        adj_matrix = target_antivectorize(target_sample)
        target_data_as_adj_matrices.append(adj_matrix)

    return np.array(target_data_as_adj_matrices).reshape((len(target_data_as_adj_matrices), N_TARGET_NODES, N_TARGET_NODES, 1))


# Taken from https://github.com/basiralab/DGN/blob/master/model.py
# Create data objects  for the DGN
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
def cast_to_DGN_graph(array_of_tensors, subject_type=None, flat_mask=None):
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    dataset = []
    for mat in array_of_tensors:
        # Allocate numpy arrays
        edge_index = np.zeros((2, N_ROI * N_ROI))
        edge_attr = np.zeros((N_ROI * N_ROI, CHANNELS))
        x = np.zeros((N_ROI, 1))
        y = np.zeros((1,))

        counter = 0
        for i in range(N_ROI):
            for j in range(N_ROI):
                edge_index[:, counter] = [i, j]
                edge_attr[counter, :] = mat[i, j]
                counter += 1

        # Fill node feature matrix (no features every node is 1)
        for i in range(N_ROI):
            x[i, 0] = 1

        # Get graph labels
        y[0] = None

        if flat_mask is not None:
            edge_index_masked = []
            edge_attr_masked = []
            for i, val in enumerate(flat_mask):
                if val == 1:
                    edge_index_masked.append(edge_index[:, i])
                    edge_attr_masked.append(edge_attr[i, :])
            edge_index = np.array(edge_index_masked).T
            edge_attr = edge_attr_masked

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        con_mat = torch.tensor(mat, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    con_mat=con_mat,  y=y, label=subject_type)
        dataset.append(data)
    return dataset
