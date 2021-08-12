# Train on simulated data (S: normal random dist.) or external data (E)
DATASET = "S"

# Path to external source data (binary file in NumPy .npy format)
# Dimension of source data: [N_SUBJECTS, N_SOURCE_NODES, N_SOURCE_NODES]
# Ignored if DATASET = "S"
SOURCE_DATA_PATH = "/path/to/external/source/data"

# Path to external target data (binary file in NumPy .npy format)
# Dimension of target data: [N_SUBJECTS, N_TARGET_NODES, N_TARGET_NODES]
# Ignored if DATASET = "S"
TARGET_DATA_PATH = "/path/to/external/target/data"

# Number of subjects in simulated data (Overwritten if DATASET = "E")
N_SUBJECTS = 100

# Number of ROIs in source brain graph for simulated data (Overwritten if DATASET = "E")
N_SOURCE_NODES = 35

# Number of ROIs in target brain graph for simulated data (Overwritten if DATASET = "E")
N_TARGET_NODES = 160

# Number of traning epochs
N_EPOCHS = 400

# Number of folds for cross validation
N_FOLDS = 3

# Number of clusters
# SIMLR splits source and target data into N_CLUSTERS amount of clusters
N_CLUSTERS = 3

# Model name
MODEL_NAME = "TIS_Net_test_2"

# DGN parameters for source domain model
DGN_MODEL_PARAMS_SOURCE = {
    "N_ROIs": N_SOURCE_NODES,
    "n_attr": 1,
    "Linear1": {"in": 1, "out": 36},
    "conv1": {"in": 1, "out": 36},

    "Linear2": {"in": 1, "out": 36*24},
    "conv2": {"in": 36, "out": 24},

    "Linear3": {"in": 1, "out": 24*5},
    "conv3": {"in": 24, "out": 5}
}

# DGN parameters for target domain model
DGN_MODEL_PARAMS_TARGET = {
    "N_ROIs": N_TARGET_NODES,
    "n_attr": 1,
    "Linear1": {"in": 1, "out": 36},
    "conv1": {"in": 1, "out": 36},

    "Linear2": {"in": 1, "out": 36*24},
    "conv2": {"in": 36, "out": 24},

    "Linear3": {"in": 1, "out": 24*5},
    "conv3": {"in": 24, "out": 5}
}


#################################################################################
############################# DO NOT MODIFY BELOW ###############################
#################################################################################

from helper import create_simulated_data
import numpy as np

if DATASET.lower() == "e":
    X_s = np.load(SOURCE_DATA_PATH)
    X_t = np.load(TARGET_DATA_PATH)

    if X_s.shape[0] != X_t.shape[0]:
        raise ValueError(
            "Source and target datasets must have the same subject amount")

    N_SUBJECTS = X_s.shape[0]
    N_SOURCE_NODES = X_s.shape[1]
    N_TARGET_NODES = X_t.shape[2]

elif DATASET.lower() == "s":
    X_s, X_t = create_simulated_data(
        N_SUBJECTS, N_SOURCE_NODES, N_TARGET_NODES)

else:
    raise ValueError("Dataset options are E or S")
