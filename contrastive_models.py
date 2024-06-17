# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import time

from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.cluster import adjusted_rand_score

# Determine the device to use for Torch operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Function to orthogonalize a matrix using Singular Value Decomposition (SVD)
def orthogonalize(weights):
    u, _, v = torch.svd(weights)
    return torch.mm(u, v.t())

# Function to create a cluster assignment matrix for data points
def clust_matrix(X, n_clusters, centers):
    clust = None
    for i in range(n_clusters):
        d = torch.norm(X - torch.tensor(centers[i]), dim=1).reshape(-1, 1)
        if clust is None:
            clust = d
        else:
            clust = torch.cat((clust, d), dim=1)
    clust = torch.argmin(clust, axis=1)
    
    # Prepare cluster assignment matrix
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float32)
    for i in range(X.shape[0]):
        clust_assign[clust[i], i] = 1
    return clust_assign

# Function to update cluster centers
def update_centers(X, centers, clust_assign, batch_idx):
    centers = clust_assign.float() @ X.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float)
    centers_ = centers / norm
    if batch_idx == -1:
        return centers_
    return (batch_idx * centers + centers_) / (batch_idx + 1)

# Encoder class for neural network
class Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=True)
        self.linear = linear
        nn.init.orthogonal_(self.enc1.weight)

    def forward(self, x):
        x = self.enc1(x)
        if not self.linear:
            x = F.relu(x)
        return x

# Non-Contrastive Encoder class
class NonContrastive_Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(NonContrastive_Encoder, self).__init__()
        self.encoder = Encoder(in_feature, embed, linear=linear)
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def forward(self, tuple):
        anchor, positives = tuple
        encoded_anchor = self.encode(anchor)
        encoded_positives = self.encode(positives)        
        return encoded_anchor, encoded_positives
    
# Many Positive Encoder class
class ManyPosi_Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(ManyPosi_Encoder, self).__init__()
        self.encoder = Encoder(in_feature, embed, linear=linear)
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def forward(self, triplet):
        anchor, positives, negatives = triplet
        encoded_anchor = self.encode(anchor)
        encoded_positives = self.encode(positives)
        encoded_negatives = self.encode(negatives)
        return encoded_anchor, encoded_positives, encoded_negatives

# Contrastive Loss class
class contrastiveLoss(nn.Module):
    def __init__(self):
        super(contrastiveLoss, self).__init__()

    def forward(self, anchors, positives, negatives):
        assert positives.shape[1] == negatives.shape[1], "Number of negative examples and positive examples must match"
        num_of_examples = positives.shape[1]
        
        # Ensure no examples case is handled
        if num_of_examples == 0:
            return torch.tensor(0.0), torch.zeros((anchors.shape[0]), dtype=torch.float32, device=anchors.device)
        
        # Vectorized operations
        diff = negatives - positives  # Calculate (u(x^-) - u(x^+)) for all examples in the batch
        anchors = anchors.unsqueeze(1)  # Reshape anchors for broadcasting
        
        # Calculate u(x).T * (u(x^-) - u(x^+)) for all examples in the batch
        loss_per_example = torch.sum(anchors * diff, dim=-1)
        
        # Average over the number of examples for each anchor
        avg_anchor_losses = torch.mean(loss_per_example, dim=1)
        
        # Batch loss is the mean over all anchors
        batch_loss = torch.mean(avg_anchor_losses)
        return batch_loss, avg_anchor_losses

# None Loss class
class NoneLoss(nn.Module):
    def __init__(self):
        super(NoneLoss, self).__init__()

    def forward(self, anchors, positives):
        num_of_examples = positives.shape[1]
        
        # Ensure no examples case is handled
        if num_of_examples == 0:
            return torch.tensor(0.0), torch.zeros((anchors.shape[0]), dtype=torch.float32, device=anchors.device)
        
        # Vectorized operations
        anchors = anchors * -1
        anchors = anchors.unsqueeze(1)  # Reshape anchors for broadcasting
        
        # Calculate u(x).T * u(x^+) for all examples in the batch
        loss_per_example = torch.sum(anchors * positives, dim=-1)
        
        # Average over the number of examples for each anchor
        avg_anchor_losses = torch.mean(loss_per_example, dim=-1) 
        
        # Batch loss is the mean over all anchors
        batch_loss = torch.mean(avg_anchor_losses)
        return batch_loss, avg_anchor_losses

# Tensorized Contrastive Autoencoder class
class TensorizedCAE(nn.Module):
    def __init__(self, in_feature, embed, loss_f, num_tensors=2, linear=True):
        super(TensorizedCAE, self).__init__()
        self.linear = linear
        self.embed = embed
        self.CE = nn.ModuleList()
        
        # Add multiple encoders
        self.num_tensors = num_tensors
        for i in range(num_tensors):            
            self.CE.append(ManyPosi_Encoder(in_feature, embed, linear=self.linear))
        self.loss_f = loss_f     

    def forward(self, X, centers, anchor_index, batch_size):
        total_loss = torch.tensor(0, device=device, dtype=torch.float)
        embeddings = np.zeros((batch_size, self.embed))
        best_losses = np.full((batch_size,), np.inf)
        best_indices = np.full((batch_size,), -1, dtype=np.int64)

        for j in range(self.num_tensors):
            a = X[0]
            p = X[1]
            n = X[2]
            
            # Get the embeddings
            anchor, positives, negatives = self.CE[j]((a, p, n))
            
            # Calculate loss
            ls, indiv_ls = self.loss_f(anchor, positives, negatives)
            total_loss += ls
            indiv_ls = indiv_ls.detach().cpu().numpy()
            
            # Update best loss and index
            better_loss = indiv_ls < best_losses
            best_losses = np.where(better_loss, indiv_ls, best_losses)
            best_indices = np.where(better_loss, np.full_like(best_indices, j), best_indices)
            embeddings = np.where(better_loss[:, None], anchor.detach().cpu().numpy(), embeddings)
        
        # Convert total_losses to a tensor and sum
        return total_loss, embeddings, best_indices

# Tensorized Non-Contrastive Autoencoder class
class TensorizedNCAE(nn.Module):
    def __init__(self, in_feature, embed, loss_f, num_tensors=2, linear=True):
        super(TensorizedNCAE, self).__init__()
        self.linear = linear
        self.embed = embed
        self.CE = nn.ModuleList()
        
        # Add multiple encoders
        self.num_tensors = num_tensors
        for i in range(num_tensors):            
            self.CE.append(NonContrastive_Encoder(in_feature, embed, linear=self.linear))
        self.loss_f = loss_f     

    def forward(self, X, centers, anchor_index, batch_size):
        total_loss = torch.tensor(0, device=device, dtype=torch.float)
        embeddings = np.zeros((batch_size, self.embed))
        best_losses = np.full((batch_size,), np.inf)
        best_indices = np.full((batch_size,), -1, dtype=np.int64)

        for j in range(self.num_tensors):
            a = X[0]
            p = X[1]
            
            # Get the embeddings
            anchor, positives = self.CE[j]((a, p))
            
            # Calculate loss
            ls, indiv_ls = self.loss_f(anchor, positives)
            total_loss += ls
            indiv_ls = indiv_ls.detach().cpu().numpy()
            
            # Update best loss and index
            better_loss = indiv_ls < best_losses
            best_losses = np.where(better_loss, indiv_ls, best_losses)
            best_indices = np.where(better_loss, np.full_like(best_indices, j), best_indices)
            embeddings = np.where(better_loss[:, None], anchor.detach().cpu().numpy(), embeddings)
        
        # Convert total_losses to a tensor and sum
        return total_loss, embeddings, best_indices

# Training function for Non-Contrastive Autoencoder
def train_NCAE(loader, X_out, Y, method, lr=0.001, embed=2, epochs=10, batch_size=1, linear=True, CNN=False, loss_f=NoneLoss()):
    train_loss = []
    net = NonContrastive_Encoder(X_out.shape[1], embed, linear=linear).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0
        progress_bar = enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'))
        
        for batch_idx, (anchor, positives, anchor_index) in progress_bar:
            optimizer.zero_grad()
            anchor = anchor.to(device)
            positives = positives.to(device)
            anchor, positives = net((anchor, positives))
            batch_loss, indiv_ls = loss_f(anchor, positives)
            total_loss += batch_loss.item()
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
        
        # Record the loss for the epoch
        loss = total_loss / len(loader)
        train_loss.append(loss)
    
    return net, train_loss

# Training function for Tensorized Non-Contrastive Autoencoder
def train_NCTAE(loader, X_out, Y, method, n_clusters=2, n_tensors=1, lr=0.001, embed=2, epochs=10, batch_size=1, linear=True, CNN=False, loss_f=NoneLoss(), log=False):
    with torch.no_grad():
        centers, indices = kmeans_plusplus(X_out.numpy(), n_clusters=n_tensors, random_state=20)
        clust_assign = clust_matrix(X_out, n_tensors, centers)
        centers = update_centers(X_out, centers, clust_assign, -1)
    
    centers = centers.to(device)
    train_loss = []
    net = TensorizedNCAE(X_out.shape[1], embed, loss_f, num_tensors=n_tensors, linear=linear).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0
        progress_bar = enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'))
        
        for batch_idx, (anchor, positives, anchor_index) in progress_bar:
            optimizer.zero_grad()
            anchor = anchor.to(device)
            positives = positives.to(device)
            batch_loss, embeddings_, assignments = net((anchor, positives), centers, anchor_index, batch_size)
            total_loss += batch_loss.item()
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            labels = cluster(embeddings_, assignments, n_tensors, method=method)
            
            # Update cluster assignments
            for k in range(len(anchor)):
                clust_assign[:, anchor_index[k]] = 0
                clust_assign[labels[k]][anchor_index[k]] = 1
            
            # Recalculate cluster centers
            centers = update_centers(X_out, centers, clust_assign, batch_idx)
            centers = centers.to(device)
            
            for contrastive_encoder in net.CE:
                constrain_operator_norm(contrastive_encoder.encoder.enc1.weight, max_norm=1.0)
        
        if log:
            print("ARI:", adjusted_rand_score(torch.argmax(clust_assign, axis=0).cpu().detach().numpy(), Y.cpu().detach().numpy()))
        
        # Record the loss for the epoch
        loss = total_loss / len(loader)
        train_loss.append(loss)
    
    return net, train_loss, clust_assign
# Cluster assignment function
def cluster(embeddings, assignments, n_clusters, method='tensor'):
    if method == 'tensor':
        return assignments
    else:  # if method is 'kmeans'
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
        kmeans.fit(embeddings)
        return kmeans.labels_

# Training function for Contrastive Autoencoder
def train_CAE(loader, X_out, Y, method, lr=0.001, embed=2, epochs=10, batch_size=1, linear=True, CNN=False, loss_f=contrastiveLoss()):
    train_loss = []
    net = ManyPosi_Encoder(X_out.shape[1], embed, linear=linear).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0
        progress_bar = enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'))
        
        for batch_idx, (anchor, positives, negatives, anchor_index) in progress_bar:
            optimizer.zero_grad()
            anchor = anchor.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            anchor, positives, negatives = net((anchor, positives, negatives))
            batch_loss, indiv_ls = loss_f(anchor, positives, negatives)
            total_loss += batch_loss.item()
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
        
        # Record the loss for the epoch
        loss = total_loss / len(loader)
        print('epoch ', epoch + 1, ' loss ', loss)
        train_loss.append(loss)
    
    # Return the trained network and training loss history
    return net, train_loss

# Training function for Tensorized Contrastive Autoencoder
def train_CTAE(loader, X_out, Y, method, n_tensors=1, lr=0.001, embed=2, epochs=10, batch_size=1, linear=True, CNN=False, loss_f=contrastiveLoss(), log=False):
    with torch.no_grad():
        centers, indices = kmeans_plusplus(X_out.numpy(), n_clusters=n_tensors, random_state=20)
        clust_assign = clust_matrix(X_out, n_tensors, centers)
        centers = update_centers(X_out, centers, clust_assign, -1)
    
    centers = centers.to(device)
    train_loss = []
    net = TensorizedCAE(X_out.shape[1], embed, loss_f, num_tensors=n_tensors, linear=linear).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0
        progress_bar = enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'))
        
        for batch_idx, (anchor, positives, negatives, anchor_index) in progress_bar:
            optimizer.zero_grad()
            anchor = anchor.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            batch_loss, embeddings_, assignments = net((anchor, positives, negatives), centers, anchor_index, batch_size)
            total_loss += batch_loss.item()
            
            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            labels = cluster(embeddings_, assignments, n_tensors, method=method)
            
            # Update cluster assignments
            for k in range(len(anchor)):
                clust_assign[:, anchor_index[k]] = 0
                clust_assign[labels[k]][anchor_index[k]] = 1
            
            # Recalculate cluster centers
            centers = update_centers(X_out, centers, clust_assign, batch_idx)
            centers = centers.to(device)
            
            for contrastive_encoder in net.CE:
                constrain_operator_norm(contrastive_encoder.encoder.enc1.weight, max_norm=1.0)
        
        if log:
            print("ARI:", adjusted_rand_score(torch.argmax(clust_assign, axis=0).cpu().detach().numpy(), Y.cpu().detach().numpy()))
        
        # Record the loss for the epoch
        loss = total_loss / len(loader)
        if log:
            print('epoch ', epoch + 1, ' loss ', loss)
        train_loss.append(loss)
    
    # Return the trained network, training loss history, and cluster assignments
    return net, train_loss, clust_assign
# Function to run the experiment
def run_experiment(dataset_loader, X, Y, method, version, n_clusters, n_tensors, epochs, batch_size, embed, lr, model='contrastive'):
    if version == 'KMeans':
        X_embed = X.cpu().detach().numpy()  # Convert tensor to numpy
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X_embed)
        labels = kmeans.labels_
    elif version == 'CAE':
        if model == 'contrastive':
            net, loss = train_CAE(
                dataset_loader,
                X_out=X.clone(),
                Y=Y,
                method=method,
                batch_size=batch_size,
                epochs=epochs,
                embed=embed,
                lr=lr,
                loss_f=contrastiveLoss(),
                linear=True
            )
        elif model == 'non':
            net, loss = train_NCAE(
                dataset_loader,
                X_out=X.clone(),
                Y=Y,
                method=method,
                batch_size=batch_size,
                epochs=epochs,
                embed=embed,
                lr=lr,
                loss_f=NoneLoss(),
                linear=True
            )
        else:
            raise Exception("Model not implemented")
        
        X_embed = net.encode(X).cpu().detach().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_embed)
        labels = kmeans.labels_
    elif version == 'Tensor':
        if model == 'contrastive':
            net, loss, labels = train_CTAE(
                dataset_loader,
                X_out=X.clone(),
                Y=Y,
                method=method,
                batch_size=batch_size,
                epochs=epochs,
                embed=embed,
                lr=lr,
                n_tensors=n_tensors,
                loss_f=contrastiveLoss(),
                linear=True
            )
        elif model == 'non':
            net, loss, labels = train_NCTAE(
                dataset_loader,
                X_out=X.clone(),
                Y=Y,
                method=method,
                batch_size=batch_size,
                epochs=epochs,
                embed=embed,
                lr=lr,
                n_clusters=n_clusters,
                n_tensors=n_tensors,
                loss_f=NoneLoss(),
                linear=True
            )
        else:
            raise Exception("Model not implemented")
        
        labels = torch.argmax(labels, axis=0).cpu().detach().numpy()
    else:
        raise ValueError("Unsupported version")

    # Calculate and return ARI score
    return adjusted_rand_score(labels, Y.numpy())  # Convert Y to numpy if it's a tensor

# Function to normalize a tensor using Frobenius norm
def frobenius_normalize(T):
    frobenius_norm = torch.sqrt(torch.sum(T ** 2))
    normalized_T = T / frobenius_norm
    return normalized_T

# Function to compute the operator norm (largest singular value) of a matrix
def operator_norm(M):
    # Perform Singular Value Decomposition (SVD)
    U, S, V = torch.linalg.svd(M, full_matrices=False)
    # The operator norm is the largest singular value
    return S[0]

# Function to constrain the operator norm of a matrix to a maximum value
def constrain_operator_norm(M, max_norm=1.0):
    with torch.no_grad():
        norm = operator_norm(M)
        if norm > max_norm:
            M.data = M.data * (max_norm / norm)
