"""
    Main function of Template-Based Inter-modality Super-resolution of Brain Connectivity (TIS_Net) framework 
    for predicting target domain connectional brain template (CBT) from a different source domain
    
    ---------------------------------------------------------------------
    
    This file contains the driver code for the training and testing process of our TIS_Net model.
    TIS_Net class is a encapsulation of used models in our framework. See the model.py file for details.

        X_s and X_t:
            Source and target data with shape [N_SUBJECTS x  N_FEAT] where
                N_SUBJECTS is the number of subjects
                N_FEAT is the number of features for each subject.
                    N_FEAT is equal to N_ROIs * (N_ROIs - 1) / 2 where N_ROI is the number of region of interests in the source and target brain graph, respectively.
                    In other words, vectorizing the adjacency matrix of the source and target brain graph with shape [N_ROI x N_ROI] gives N_FEAT vector with shape [(N_ROIs * (N_ROIs - 1) / 2) x 1]
                    
        Clustered data:
            Source and target data are clustered into N_CLUSTERS using SIMLR in each fold seperately for training and testing sets.
            Clustered data is stored in lists with length equal to N_CLUSTERS
            Number of clusters (N_CLUSTERS) can be specified in config.py file but note that our framework is based on N_CLUSTERS = 3.

        Training algorithm:
            For each source cluster -> get CBT with source DGN model (N_CLUSTER CBTs) -> N_CLUSTER CBTs to 1 CBT for source domain using source DGN model
            For each target cluster -> get CBT with target DGN model (N_CLUSTER CBTs) -> N_CLUSTER CBTs to 1 CBT for target domain using target DGN model

            Feed source CBT into generator
            Use discriminator on generator's output and target CBT
            Guide generator
        
        
        TIS_Net.train(source_clusters_train, target_clusters_train)
                Inputs:
                        source_clusters_train, target_clusters_train: A list of clusters for the source and target domain.
                                                                      Number of clusters can be specified in config.py file but note that our method is based on 3 clusters.
                                                                      Clusters are created with SIMLR before feeding into train method
                                                                      A CBT is obtained for each cluster, then these a general CBT is obtained from these CBTs in this method
                                                                      This process is done for both source and target domain
                                                                      After that CBT is feed into generator and discriminator
                Output:
                        For each epoch, prints out L1 Loss, generator loss, discriminator loss
            
        TIS_Net.test(source_clusters_test, target_clusters_test, dgn_source_model_path, dgn_target_model_path, generator_model_path, discriminator_model_path)
                Inputs:
                        source_clusters_test, target_clusters_test: Same as in the train method with only difference these clusters are for testing
                        dgn_source_model_path, dgn_target_model_path, generator_model_path, discriminator_model_path: Paths to model files to test on given data
                Outputs:
                        Prints out the MAE, MAE of pagerank, eigenvector, betweenness centralities between predicted and ground-truth CBT.
                        Also prints out the mean norm distance between predicted CBT and each test target sample
    
    To evaluate our framework we used 3-fold cross-validation strategy.
    Number of folds can be specified in config.py file via N_FOLDS variable
    ---------------------------------------------------------------------
    Copyright 2021 Furkan Pala, Istanbul Technical University.
    All rights reserved.
"""

import os
import numpy as np
from model import TIS_Net
from sklearn.model_selection import KFold
from config import N_FOLDS, N_CLUSTERS, MODEL_NAME, X_s, X_t
from helper import cast_source_to_DGN_input, cast_target_to_DGN_input, cluster_with_SIMLR, print_final_results, save_fig, print_fold_specific_results, calc_norm_distances

kf = KFold(n_splits=N_FOLDS)

"""
    To store test results between predicted and ground-truth CBT for evaluation metrices:
        MAE: Mean Absolute Error
        PR: Pagerank Centrality
        EC: Eigenvector Centrality
        BC: Betweenness Centrality
"""
mae = []
mae_pr = []
mae_ec = []
mae_bc = []

"""
    To store mean norm distance between predicted CBT and each test subject
    for each fold
"""
mean_norm_dists = []

"""
    To store predicted and target CBTs for each fold
    Contains objects with keys "predidcted" and "ground_truth" 
"""
CBTs = []

fold = 0
for train_index, test_index in kf.split(X_s):
    print(10 * "#" + " FOLD " + str(fold) + " " + 10 * "#")

    X_train_source, X_test_source, X_train_target, X_test_target = X_s[train_index], X_s[test_index], X_t[train_index], X_t[test_index]

    # SIMLR clustering
    X_casted_train_source_clusters  = cluster_with_SIMLR(X_train_source, cast_source_to_DGN_input, N_CLUSTERS)
    X_casted_train_target_clusters  = cluster_with_SIMLR(X_train_target, cast_target_to_DGN_input, N_CLUSTERS)
    X_casted_test_source_clusters   = cluster_with_SIMLR(X_test_source,  cast_source_to_DGN_input, N_CLUSTERS)
    X_casted_test_target_clusters   = cluster_with_SIMLR(X_test_target,  cast_target_to_DGN_input, N_CLUSTERS)

    # Create an instance of TIS Net class
    TIS_net = TIS_Net()

    # Train on given source and target training clusters
    TIS_net.train(X_casted_train_source_clusters,
                  X_casted_train_target_clusters)

    # Change the paths below if you want to load an external model
    model_path = f"models/{MODEL_NAME}/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dgn_src_model_path = f"{model_path}/dgn_src_fold_{fold}.model"
    dgn_target_model_path = f"{model_path}/dgn_target_fold_{fold}.model"
    generator_model_path = f"{model_path}/generator_fold_{fold}.model"
    discriminator_model_path = f"{model_path}/discriminator_fold_{fold}.model"

    # Save the weights of trained models
    TIS_net.save_models(dgn_src_model_path, dgn_target_model_path, generator_model_path, discriminator_model_path)

    # Test on given source and target testing clusters with given models
    predicted_CBT, ground_truth_CBT, evaluation_results_fold_specific = TIS_net.test(
        X_casted_test_source_clusters,
        X_casted_test_target_clusters,
        dgn_src_model_path,
        dgn_target_model_path,
        generator_model_path,
        discriminator_model_path
    )

    # Calculate norm distances between predicted CBT and each test subject
    mean_norm_distance_fold_specific = np.mean(calc_norm_distances(predicted_CBT, X_test_target))

    # Print results for each fold on stdout
    print_fold_specific_results(fold, evaluation_results_fold_specific, mean_norm_distance_fold_specific)

    # Save figures of predicted and ground-truth CBTs in "output/{MODEL_NAME}" folder
    save_fig(predicted_CBT, f"Predicted CBT for fold {fold}", f"pred_CBT_fold_{fold}.png")
    save_fig(ground_truth_CBT, f"Ground-truth CBT for fold {fold}", f"gt_CBT_fold_{fold}.png")
    
    # Save residual figure of predicted and ground-truth CBTs in "output/{MODEL_NAME}" folder
    save_fig(np.abs(predicted_CBT - ground_truth_CBT), f"Residual for fold {fold}", f"residual_fold_{fold}.png")

    mae.append(evaluation_results_fold_specific["MAE"].item())
    mae_pr.append(evaluation_results_fold_specific["MAE(PR)"].item())
    mae_ec.append(evaluation_results_fold_specific["MAE(EC)"].item())
    mae_bc.append(evaluation_results_fold_specific["MAE(BC)"].item())


    mean_norm_dists.append(mean_norm_distance_fold_specific)

    CBTs.append({
        "predicted" : predicted_CBT,
        "ground_truth" : ground_truth_CBT
    })

    fold += 1

# Print results across folds on stdout
print_final_results(mae, mae_pr, mae_ec, mae_bc, mean_norm_dists)