import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import numpy as np
from config import N_CLUSTERS, N_SOURCE_NODES, N_TARGET_NODES, DGN_MODEL_PARAMS_SOURCE, DGN_MODEL_PARAMS_TARGET, N_EPOCHS, N_CLUSTERS
from helper import cast_to_DGN_graph, source_to_graph
from random import sample
from centrality import topological_measures
import time
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        nn = Sequential(Linear(1, N_SOURCE_NODES * N_SOURCE_NODES), ReLU())
        self.conv1 = NNConv(N_SOURCE_NODES, N_SOURCE_NODES, nn, aggr='mean',
                            root_weight=True, bias=True)
        self.conv11 = BatchNorm(
            N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES * N_TARGET_NODES), ReLU())
        self.conv2 = NNConv(N_SOURCE_NODES, N_TARGET_NODES, nn, aggr='mean',
                            root_weight=True, bias=True)
        self.conv22 = BatchNorm(
            N_TARGET_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES * N_TARGET_NODES), ReLU())
        self.conv3 = NNConv(N_SOURCE_NODES, N_TARGET_NODES, nn, aggr='mean',
                            root_weight=True, bias=True)
        self.conv33 = BatchNorm(
            N_TARGET_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = torch.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        """
            If the target data includes many ROIs,
            enabling the second layer below makes the third layer consumed too much memory that program crashes
        """
        
        #Enabled
        # x2 = torch.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        # x2 = F.dropout(x2, training=self.training)
        #Enabled End
        
        x3 = torch.sigmoid(self.conv33(self.conv3(x1, edge_index, edge_attr)))
        x3 = F.dropout(x3, training=self.training)

        x4 = torch.matmul(x3.t(), x3)

        x4 = x4 / torch.max(x4)

        ind = np.diag_indices(x4.shape[0])
        x4[ind[0], ind[1]] = torch.ones(
            x4.shape[0]).to(x4.device)

        return x4


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(N_TARGET_NODES, N_TARGET_NODES)

        self.conv2 = GCNConv(N_TARGET_NODES, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = x.unsqueeze(0)
        edge_attr = edge_attr.squeeze()

        x = torch.sigmoid(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)

        x = torch.sigmoid(self.conv2(x, edge_index, edge_attr))

        x = x.squeeze(0)

        return x


class DGN(torch.nn.Module):
    def __init__(self, MODEL_PARAMS):
        super(DGN, self).__init__()
        self.model_params = MODEL_PARAMS

        nn = Sequential(Linear(
            self.model_params["Linear1"]["in"], self.model_params["Linear1"]["out"]), ReLU())
        self.conv1 = NNConv(
            self.model_params["conv1"]["in"], self.model_params["conv1"]["out"], nn, aggr='mean')

        nn = Sequential(Linear(
            self.model_params["Linear2"]["in"], self.model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(
            self.model_params["conv2"]["in"], self.model_params["conv2"]["out"], nn, aggr='mean')

        nn = Sequential(Linear(
            self.model_params["Linear3"]["in"], self.model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(
            self.model_params["conv3"]["in"], self.model_params["conv3"]["out"], nn, aggr='mean')

    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format) 


        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = F.relu(self.conv2(x, edge_index, edge_attr))

        x = F.relu(self.conv3(x, edge_index, edge_attr))

        repeated_out = x.repeat(self.model_params["N_ROIs"], 1, 1)
        repeated_t = torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)

        return cbt


class TIS_Net():
    def __init__(self):
        # Use cuda if avaliable
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on cuda")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        """
            TIS_Net consists of 4 architectures
            - Generator:     Performs mapping from source domain to target domain
            - Discriminator: Works with generator in an adversarial manner, 
                             guides generator to predict outputs which are indistinguishable from ground-truth samples 
            - DGN Source:    Predicts a CBT for source domain
            - DGN Target:    Predicts a CBT for target domain
        """
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.dgn_source = DGN(DGN_MODEL_PARAMS_SOURCE).to(self.device)
        self.dgn_target = DGN(DGN_MODEL_PARAMS_TARGET).to(self.device)

        # Initalize loss functions
        self.l1_loss = torch.nn.L1Loss().to(self.device)
        self.adversarial_loss = torch.nn.BCELoss().to(self.device)

        # Initialize optimizers
        self.generator_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=0.025, betas=(0.5, 0.999))

        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=0.025, betas=(0.5, 0.999))

        self.dgn_source_optimizer = torch.optim.AdamW(
            self.dgn_source.parameters(), lr=0.005, weight_decay=0.00)

        self.dgn_target_optimizer = torch.optim.AdamW(
            self.dgn_target.parameters(), lr=0.005, weight_decay=0.00)

        self.l1_losses_test_fold_specific = []
        self.pagerank_losses_test_fold_specific = []
        self.betweenness_losses_test_fold_specific = []
        self.eigenvector_losses_test_fold_specific = []


    def train(self, X_casted_train_source_clusters, X_casted_train_target_clusters):
        # Preprocess data for source DGN
        # Taken from https://github.com/basiralab/DGN/blob/master/model.py
        train_src_mean = [np.mean(train_src, axis=(0, 1, 2))
                          for train_src in X_casted_train_source_clusters]

        loss_src_weightes = []

        for i in range(N_CLUSTERS):
            loss_src_weightes.append(torch.tensor(np.array(list(
                (1 / train_src_mean[i]) / np.max(1 / train_src_mean[i]))*len(X_casted_train_source_clusters[i])), dtype=torch.float32))

        loss_src_weightes = [i.to(self.device) for i in loss_src_weightes]

        train_src_casted = [[d.to(self.device) for d in cast_to_DGN_graph(train_src)]
                            for train_src in X_casted_train_source_clusters]

        targets_src = [[torch.tensor(tensor, dtype=torch.float32).to(
            self.device) for tensor in train_src] for train_src in X_casted_train_source_clusters]

        # Preprocess data for target DGN
        # Taken from https://github.com/basiralab/DGN/blob/master/model.py
        train_target_mean = [np.mean(train_target, axis=(
            0, 1, 2)) for train_target in X_casted_train_target_clusters]

        loss_target_weightes = []

        for i in range(N_CLUSTERS):
            loss_target_weightes.append(torch.tensor(np.array(list(
                (1 / train_target_mean[i]) / np.max(1 / train_target_mean[i]))*len(X_casted_train_target_clusters[i])), dtype=torch.float32))

        loss_target_weightes = [i.to(self.device)
                                for i in loss_target_weightes]

        train_target_casted = [[d.to(self.device) for d in cast_to_DGN_graph(
            train_target)] for train_target in X_casted_train_target_clusters]

        targets_target = [[torch.tensor(tensor, dtype=torch.float32).to(
            self.device) for tensor in train_target] for train_target in X_casted_train_target_clusters]

        # Training loop
        for epoch in range(N_EPOCHS):
            start = time.time() # Modification
            with torch.autograd.set_detect_anomaly(True):
                # Lists to store CBTs for each cluster
                source_CBTs_for_clusters = []
                target_CBTs_for_clusters = []

                source_cluster_to_cbt_loss = []
                target_cluster_to_cbt_loss = []

                # Get cluster specific CBTs
                for i in range(N_CLUSTERS):
                    # Get source CBT for each cluster
                    losses_src, final_source_cbt = self.get_CBT(
                        train_src_casted[i], targets_src[i], loss_src_weightes[i], self.dgn_source, N_SOURCE_NODES)

                    source_cluster_to_cbt_loss.append(
                        torch.mean(torch.stack(losses_src)))

                    final_source_cbt = final_source_cbt.detach().cpu().clone().numpy()
                    source_CBTs_for_clusters.append(final_source_cbt)

                    # Get target CBT for each cluster
                    losses_target, final_target_cbt = self.get_CBT(
                        train_target_casted[i], targets_target[i], loss_target_weightes[i], self.dgn_target, N_TARGET_NODES)

                    target_cluster_to_cbt_loss.append(
                        torch.mean(torch.stack(losses_target)))

                    final_target_cbt = final_target_cbt.detach().cpu().clone().numpy()
                    target_CBTs_for_clusters.append(final_target_cbt)

                # Original
                #source_CBTs_for_clusters = np.array(
                    #source_CBTs_for_clusters).reshape(N_CLUSTERS, 35, 35, 1) #Is There a mistake? 
                #target_CBTs_for_clusters = np.array(
                    #target_CBTs_for_clusters).reshape(N_CLUSTERS, 160, 160, 1) # Static?

                ### Yekta's fix ###
                source_CBTs_for_clusters = np.array(
                    source_CBTs_for_clusters).reshape(N_CLUSTERS, N_SOURCE_NODES, N_SOURCE_NODES, 1) # Proposed Fix
                target_CBTs_for_clusters = np.array( 
                    target_CBTs_for_clusters).reshape(N_CLUSTERS, N_TARGET_NODES, N_TARGET_NODES, 1) # Proposed Fix
                ### Yekta's fix END###


                # Cluster specific CBTs to single CBT

                # Source CBTs -> CBT
                inner_train_src_mean = np.mean(
                    source_CBTs_for_clusters, axis=(0, 1, 2))

                inner_loss_src_weightes = torch.tensor(np.array(list(
                    (1 / inner_train_src_mean) / np.max(1 / inner_train_src_mean))*len(source_CBTs_for_clusters)), dtype=torch.float32)

                inner_loss_src_weightes = inner_loss_src_weightes.to(
                    self.device)

                inner_train_src_casted = [
                    d.to(self.device) for d in cast_to_DGN_graph(source_CBTs_for_clusters)]

                inner_targets_src = [torch.tensor(tensor, dtype=torch.float32).to(
                    self.device) for tensor in source_CBTs_for_clusters]

                losses_src, final_source_cbt = self.get_CBT(
                    inner_train_src_casted, inner_targets_src, inner_loss_src_weightes, self.dgn_source, N_SOURCE_NODES)

                losses_src.append(torch.mean(
                    torch.stack(source_cluster_to_cbt_loss)))

                # Target CBTs -> CBT
                inner_train_target_mean = np.mean(
                    target_CBTs_for_clusters, axis=(0, 1, 2))

                inner_loss_target_weightes = torch.tensor(np.array(list(
                    (1 / inner_train_target_mean) / np.max(1 / inner_train_target_mean))*len(target_CBTs_for_clusters)), dtype=torch.float32)

                inner_loss_target_weightes = inner_loss_target_weightes.to(
                    self.device)

                inner_train_target_casted = [
                    d.to(self.device) for d in cast_to_DGN_graph(target_CBTs_for_clusters)]

                inner_targets_target = [torch.tensor(tensor, dtype=torch.float32).to(
                    self.device) for tensor in target_CBTs_for_clusters]

                losses_target, final_target_cbt = self.get_CBT(
                    inner_train_target_casted, inner_targets_target, inner_loss_target_weightes, self.dgn_target, N_TARGET_NODES)

                losses_target.append(torch.mean(
                    torch.stack(target_cluster_to_cbt_loss)))

                # Convert source CBT to torch.geometric.Data object
                row0 = np.repeat(np.arange(N_SOURCE_NODES), N_SOURCE_NODES)
                row1 = np.tile(np.arange(N_SOURCE_NODES), N_SOURCE_NODES)
                edge_index = np.array([row0, row1])
                edge_index_tensor = torch.from_numpy(edge_index).long()

                data_source = Data(x=final_source_cbt, edge_index=edge_index_tensor,
                                   edge_attr=final_source_cbt.view(-1, 1)).to(self.device)

                # Convert target CBT to torch.geometric.Data object
                row0 = np.repeat(np.arange(N_TARGET_NODES), N_TARGET_NODES) # changed from 160,160 N_TARGET_NODES,N_TARGET_NODES
                row1 = np.tile(np.arange(N_TARGET_NODES), N_TARGET_NODES) # changed from 160,160 N_TARGET_NODES,N_TARGET_NODES
                edge_index = np.array([row0, row1])
                edge_index_tensor = torch.from_numpy(edge_index).long()

                data_target = Data(x=final_target_cbt, edge_index=edge_index_tensor,
                                   edge_attr=final_target_cbt.view(-1, 1)).to(self.device)

                # generator_loss = []
                # discriminator_loss = []
                # L1_losses = []

                gen_out = self.generator(data_source)

                # Calculate L1 Loss
                L1_loss = self.l1_loss(data_target.x, gen_out)
                # L1_losses.append(l1)

                # Calculate pearson coor
                vx = data_target.x - torch.mean(data_target.x)
                vy = gen_out - torch.mean(gen_out)
                cost = torch.sum(
                    vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

                pc_loss = 1 - cost

                edge_attr = gen_out.view(-1, 1)

                row0 = np.repeat(np.arange(N_TARGET_NODES), N_TARGET_NODES) # changed from 160,160 N_TARGET_NODES,N_TARGET_NODES
                row1 = np.tile(np.arange(N_TARGET_NODES), N_TARGET_NODES) # changed from 160,160 N_TARGET_NODES,N_TARGET_NODES
                edge_index = np.array([row0, row1])
                edge_index_tensor = torch.from_numpy(edge_index).long()

                gen_out_graph = Data(
                    x=gen_out, edge_index=edge_index_tensor, edge_attr=edge_attr).to(self.device)

                D_real = self.discriminator(data_target)
                D_fake = self.discriminator(gen_out_graph)

                G_adversarial = self.adversarial_loss(
                    D_fake, torch.ones_like(D_fake, requires_grad=False))

                G_loss = G_adversarial + pc_loss + L1_loss

                # generator_loss.append(G_loss)

                D_real_loss = self.adversarial_loss(
                    D_real, torch.ones_like(D_real, requires_grad=False))

                D_fake_loss = self.adversarial_loss(
                    D_fake.detach(), torch.zeros_like(D_fake, requires_grad=False))

                D_loss = (D_real_loss + D_fake_loss) / 2

                # discriminator_loss.append(D_loss)

                self.dgn_source_optimizer.zero_grad()
                dgn_src_loss = torch.mean(torch.stack(losses_src))
                dgn_src_loss.backward()
                self.dgn_source_optimizer.step()

                self.dgn_target_optimizer.zero_grad()
                dgn_target_loss = torch.mean(torch.stack(losses_target))
                dgn_target_loss.backward()
                self.dgn_target_optimizer.step()

                self.generator_optimizer.zero_grad()
                # generator_loss = torch.mean(torch.stack(generator_loss))
                G_loss.backward(retain_graph=True)
                self.generator_optimizer.step()

                self.discriminator_optimizer.zero_grad()
                # discriminator_loss = torch.mean(torch.stack(discriminator_loss))
                D_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                # L1_losses = torch.mean(torch.stack(L1_losses))
            elaplasedTime = time.time() - start
            print("[Epoch: %d]| [Discriminator loss: %f]| [Generator loss: %f], | [L1 loss: %f] | [Elapsed Time: %f] " % (
                epoch, D_loss.item(), G_loss.item(), L1_loss.item(),elaplasedTime))

    # Taken from https://github.com/basiralab/DGN/blob/master/model.py
    def get_CBT(self, train_casted, targets, loss_weights, dgn, n_nodes):
        CBTs = []
        losses = []
        for data in train_casted:
            # Compose Dissimilarity matrix from network outputs
            cbt = dgn(data)
            CBTs.append(np.array(cbt.cpu().detach()))
            views_sampled = sample(targets, 1)
            sampled_targets = torch.cat(
                views_sampled, axis=2).permute((2, 1, 0))
            expanded_cbt = cbt.expand(
                (sampled_targets.shape[0], n_nodes, n_nodes))
            # Absolute difference
            diff = torch.abs(expanded_cbt - sampled_targets)
            sum_of_all = torch.mul(diff, diff).sum(
                axis=(1, 2))  # Sum of squares
            l = torch.sqrt(sum_of_all)  # Square root of the sum
            losses.append((l * loss_weights[:1]).sum())

        final_cbt = torch.tensor(
            np.median(CBTs, axis=0), dtype=torch.float32).to(self.device)

        return losses, final_cbt

    def test(self, X_casted_test_source_clusters, X_casted_test_target_clusters, 
            dgn_src_model_path, dgn_target_model_path, generator_model_path, discriminator_model_path):

        # Load models
        self.dgn_source.load_state_dict(torch.load(dgn_src_model_path))
        self.dgn_target.load_state_dict(torch.load(dgn_target_model_path))
        self.generator.load_state_dict(torch.load(generator_model_path))
        self.discriminator.load_state_dict(torch.load(discriminator_model_path))

        self.dgn_source.eval()
        self.dgn_target.eval()
        self.generator.eval()

        # Get source CBT
        src_CBTs = []
        target_CBTs = []

        for i in range(N_CLUSTERS):
            test_src_casted = [d.to(self.device) for d in cast_to_DGN_graph(
                X_casted_test_source_clusters[i])]

            source_cbts_test = []

            test_src_data = [d.to(self.device) for d in test_src_casted]

            for data in test_src_data:
                cbt = self.dgn_source(data)
                source_cbts_test.append(np.array(cbt.cpu().detach()))

            cluster_spefific_source_cbt = torch.tensor(np.median(source_cbts_test, axis=0),
                            dtype=torch.float32).to(self.device)
            cluster_spefific_source_cbt = cluster_spefific_source_cbt.detach().cpu().clone().numpy()
            src_CBTs.append(cluster_spefific_source_cbt)

            test_target_casted = [d.to(self.device) for d in cast_to_DGN_graph(
                X_casted_test_target_clusters[i])]

            target_cbts_test = []

            test_target_data = [d.to(self.device) for d in test_target_casted]
            for data in test_target_data:
                cbt = self.dgn_target(data)
                target_cbts_test.append(np.array(cbt.cpu().detach()))

            cluster_spefific_target_cbt = torch.tensor(np.median(target_cbts_test, axis=0),
                            dtype=torch.float32).to(self.device)
            cluster_spefific_target_cbt = cluster_spefific_target_cbt.detach().cpu().clone().numpy()

            target_CBTs.append(cluster_spefific_target_cbt)

        src_CBTs = np.array(src_CBTs).reshape(N_CLUSTERS, N_SOURCE_NODES, N_SOURCE_NODES, 1) # changed from 35,35 N_SOURCE_NODES,N_SOURCE_NODES
        target_CBTs = np.array(target_CBTs).reshape(N_CLUSTERS, N_TARGET_NODES, N_TARGET_NODES, 1) # changed from 165,165 N_TARGET_NODES,N_TARGET_NODES

        test_src_casted = [d.to(self.device) for d in cast_to_DGN_graph(src_CBTs)]

        source_cbts_test = []

        test_src_data = [d.to(self.device) for d in test_src_casted]
        for data in test_src_data:
            cbt = self.dgn_source(data)
            source_cbts_test.append(np.array(cbt.cpu().detach()))

        final_src_test_cbt = torch.tensor(
            np.median(source_cbts_test, axis=0), dtype=torch.float32).to(self.device)

        # Get target CBT
        test_target_casted = [d.to(self.device) for d in cast_to_DGN_graph(target_CBTs)]

        target_cbts_test = []

        test_target_data = [d.to(self.device) for d in test_target_casted]
        for data in test_target_data:
            cbt = self.dgn_target(data)
            target_cbts_test.append(np.array(cbt.cpu().detach()))

        final_target_test_cbt = torch.tensor(
            np.median(target_cbts_test, axis=0), dtype=torch.float32).to(self.device)

        final_src_test_cbt = final_src_test_cbt.detach().cpu().clone().numpy()

        data_source = source_to_graph(final_src_test_cbt).to(self.device)

        G_output_test = self.generator(data_source)

        L1_loss_test = self.l1_loss(G_output_test, final_target_test_cbt)

        predicted_CBT = G_output_test.detach().cpu().clone().numpy()
        ground_truth_CBT = final_target_test_cbt.detach().cpu().clone().numpy()

        fake_topology_test = topological_measures(predicted_CBT)
        real_topology_test = topological_measures(ground_truth_CBT)

        pagerank_loss_test = self.l1_loss(fake_topology_test[0], real_topology_test[0])
        betweenness_loss_test = self.l1_loss(fake_topology_test[1], real_topology_test[1])
        eigenvector_loss_test = self.l1_loss(fake_topology_test[2], real_topology_test[2])

        self.pagerank_losses_test_fold_specific.append(pagerank_loss_test.detach().cpu().numpy())
        self.betweenness_losses_test_fold_specific.append(betweenness_loss_test.detach().cpu().numpy())
        self.eigenvector_losses_test_fold_specific.append(eigenvector_loss_test.detach().cpu().numpy())
        self.l1_losses_test_fold_specific.append(L1_loss_test.detach().cpu().numpy())

        evaluation_results_fold_specific = {
            "MAE" : L1_loss_test,
            "MAE(PR)" : pagerank_loss_test,
            "MAE(EC)" : eigenvector_loss_test,
            "MAE(BC)" : betweenness_loss_test,
        }

        return predicted_CBT, ground_truth_CBT, evaluation_results_fold_specific

    
    def save_models(self, dgn_src_model_path, dgn_target_model_path, generator_model_path, discriminator_model_path):
        torch.save(self.dgn_source.state_dict(),    dgn_src_model_path)
        torch.save(self.dgn_target.state_dict(),    dgn_target_model_path)
        torch.save(self.generator.state_dict(),     generator_model_path)
        torch.save(self.discriminator.state_dict(), discriminator_model_path)
