import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics.cluster import adjusted_rand_score
import os
import math

from data import parallel_line, orthogonal, triangle, lines_3D, real_data_loader, add_noise_data
device = 'cpu' if torch.backends.mps.is_available() else 'cpu'


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

class Autoencoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Autoencoder, self).__init__()
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=False)
        self.dec1 = nn.Linear(in_features=embed, out_features=in_feature, bias=False)
        # nn.init.orthogonal_(self.enc1.weight)
        self.linear = linear
    def forward(self, x):
        x = self.enc1(x)
        if self.linear == False:
            x = F.relu(x)
        x = self.dec1(x)
        return x
class Encoder(nn.Module):
    def __init__(self, in_feature, embed, linear=True):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(in_features=in_feature, out_features=embed, bias=False)
        self.linear = linear
    def forward(self, x):
        x = self.enc1(x).to(device)
        if not self.linear:
            x = F.relu(x).to(device)
        return x

class Decoder(nn.Module):
    def __init__(self, embed, out_feature):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=embed, out_features=out_feature, bias=False).to(device)
    def forward(self, x):
        x = self.dec1(x).to(device)
        return x

class CNN_Autoencoder(nn.Module):
    def __init__(self,embed):
        super(CNN_Autoencoder,self).__init__()
        self.enc1 = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, embed),
            # 10
            # nn.Softmax(),
            )
        self.dec1 = nn.Sequential(
            # 10
            nn.Linear(embed, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )
    def forward(self, x):
        enc = self.enc1(x)
        # print("encoded shape", enc.shape())
        dec = self.dec1(enc)
        return dec

class TensorizedAE(nn.Module):
    def __init__(self, n_tensors, in_feature, embed, linear=True, CNN=False):
        # Class representing a tensorized architecture for autoencoders
        super(TensorizedAE, self).__init__()
        self.n_tensors = n_tensors
        self.linear = linear
        self.AEs = nn.ModuleList()
        for i in range(n_tensors):
            if CNN:
                self.AEs.append(CNN_Autoencoder(embed))
            else:
                self.AEs.append(Autoencoder(in_feature, embed, linear))
        
    def forward(self, X, centers, X_out=None):
        # Enters input to each tensor, and returns the embedding and the reconstruction
        batch_size=X.shape[0]
        if X_out==None:
            X_out=X
        embeddings = []
        reconstructed = []
        for j in range(self.n_tensors):
            x = X - centers[j]
            x_out = X_out - centers[j]
            
            embed = self.AEs[j].enc1(x)
            decod = self.AEs[j].dec1(embed)
            embeddings.append(embed)
            reconstructed.append(decod)
        return embeddings, reconstructed
    def orthogonalize(self):
        with torch.no_grad():
            for i in range(self.n_tensors):
                original_shape = self.AEs[i].enc1.weight.data.shape
                q, r = torch.qr(self.AEs[i].enc1.weight.data)
                self.AEs[i].enc1.weight.data = q[:, :original_shape[1]]

class TensorizedAEloss(nn.Module):
    # Calculates the loss for a tensorized structure. Will return the sum of losses across
    # tensors, and the embeddings of the tensors that returned the best loss
    def __init__(self, in_feature, reg, n_tensors, linear=True, CNN=False):
        super(TensorizedAEloss, self).__init__()
        self.n_tensors = n_tensors
        self.reg = reg
        self.CNN = CNN
    def forward(self, embeddings, reconstructed, x_out, centers):
        batch_size=embeddings[0].shape[0]
        total_loss = torch.zeros(batch_size,device=device)
        best_losses = np.full(batch_size, np.inf)
        best_indices = np.full(batch_size, -1, dtype=np.int64)
        for i in range(self.n_tensors):
            indiv_losses = F.mse_loss(reconstructed[i], x_out-centers[i], reduction='none').mean(dim=1) + (self.reg * torch.square(torch.norm(embeddings[i])))
            numpy_loss = indiv_losses.cpu().detach().numpy()
            total_loss += indiv_losses
            better_loss = numpy_loss < best_losses
            best_losses[better_loss] = numpy_loss[better_loss]
            best_indices[better_loss] = i  # Assuming i is the current index or identifier in your loop
        return total_loss, best_indices

def train_batch_TAE(X,Y, embed, n_tensors=2, lr=0.1, reg=0, epochs=100,
              number_of_batches=1, linear=True, CNN=False, X_out=None, printing=False):
    if X_out == None:
        X_out = X
    with torch.no_grad():
        centers, indices = kmeans_plusplus(X_out.numpy(), n_clusters=n_tensors, random_state=20)
        clust_assign = clust_matrix(X_out, n_tensors, centers)
        centers = clust_assign.float() @ X_out.float()
        norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X_out.shape[1],dtype=torch.float)
        centers = centers / norm    
    X = X.to(device)
    Y = Y.to(device)
    X_out = X_out.to(device)
    clust_assign = clust_assign.to(device)
    centers = centers.to(device)
    net = TensorizedAE(n_tensors=n_tensors, in_feature=X.shape[1], embed=embed, linear=linear, CNN=CNN).to(device)
    loss_f = TensorizedAEloss(in_feature=X.shape[1], reg=reg, n_tensors=n_tensors, linear=linear, CNN=CNN)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    total_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, reconstructed = net(X, centers, X_out)
        epoch_loss, assignments = loss_f(embeddings, reconstructed, X_out, centers)
        epoch_loss = epoch_loss.sum(dim=0)/X.shape[0]
        total_loss.append(epoch_loss.item())
        epoch_loss.backward()
        optimizer.step()
        # net.orthogonalize()
        for k in range(len(assignments)):
                clust_assign[:, k] = 0
                clust_assign[assignments[k]][k] = 1
        new_centers = clust_assign @ X_out
        new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1],dtype=torch.float).to(device)
        centers = new_centers / new_norm
        if printing:
            print('epoch ', epoch, ' loss ', epoch_loss)
        # elif (epoch + 1) == epochs:
        #     print('epoch ', epoch, ' loss ', epoch_loss)
        
    best_embeddings = torch.stack([net.AEs[assignments[i]].enc1(X[i]) for i in range(X.shape[0])], dim=0)
    reconstrctions = torch.stack([net.AEs[assignments[i]](X[i]-centers[assignments[i]]) + centers[assignments[i]] for i in range(X.shape[0])], dim=0)
    assignments = torch.argmax(clust_assign,axis=0).numpy()
    return best_embeddings, assignments, total_loss, reconstrctions


def train_batch_AE(net, X, Y, lr=0.1, epochs=100, X_out=None):
    if X_out == None:
        X_out = X

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.1)
    X = X.to(device)
    Y = Y.to(device)
    X_out = X_out.to(device)
    train_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, X_out)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
    return train_loss



# First center, then encode
class SharedOne(nn.Module):
    def __init__(self, n_tensors, in_feature, fst_embed, snd_embed, linear=True, CNN=False):
        # Class representing a tensorized architecture for autoencoders
        super(SharedOne, self).__init__()
        self.n_tensors = n_tensors
        self.linear = linear
        self.shared_encoder = Encoder(in_feature, fst_embed, linear)
        self.shared_decoder = Decoder(fst_embed, in_feature)
        self.AEs = nn.ModuleList()
        for i in range(n_tensors):
            self.AEs.append(Autoencoder(fst_embed, snd_embed, linear))
        
    def forward(self, X, centers, X_out=None):
        # Enters input to each tensor, and returns the embedding and the reconstruction
        batch_size=X.shape[0]
        if X_out==None:
            X_out=X
        embeddings = []
        reconstructed = []
        for j in range(self.n_tensors):
            x = X - centers[j]
            x_out = X_out - centers[j]
            reduced = self.shared_encoder(x)
            embed = self.AEs[j].enc1(reduced)
            fst_decod = self.AEs[j].dec1(embed)
            snd_decod = self.shared_decoder(fst_decod)
            embeddings.append(embed)
            reconstructed.append(snd_decod)
        return embeddings, reconstructed
    
# Encode X AND the centers with the shared Encoder, then do tensorization
class SharedTwo(nn.Module):
    def __init__(self, n_tensors, in_feature, fst_embed, snd_embed, linear=True, CNN=False):
        # Class representing a tensorized architecture for autoencoders
        super(SharedTwo, self).__init__()
        self.n_tensors = n_tensors
        self.linear = linear
        self.shared_encoder = Encoder(in_feature, fst_embed, linear)
        self.shared_decoder = Decoder(fst_embed, in_feature)
        self.AEs = nn.ModuleList()
        for i in range(n_tensors):
            self.AEs.append(Autoencoder(fst_embed, snd_embed, linear))
        
    def forward(self, X, centers, X_out=None):
        # Enters input to each tensor, and returns the embedding and the reconstruction
        batch_size=X.shape[0]
        if X_out==None:
            X_out=X
        embeddings = []
        reconstructed = []
        uncentered = []
        reduced_x = self.shared_encoder(X)
        reduced_x_out = self.shared_encoder(X_out)
        reduced_centers = self.shared_encoder(centers)
        for j in range(self.n_tensors):
            x = reduced_x - reduced_centers[j]
            x_out = reduced_x_out - reduced_centers[j]
            embed = self.AEs[j].enc1(x)
            decod = self.AEs[j].dec1(embed)
            embeddings.append(embed)
            reconstructed.append(self.shared_decoder(decod))
            uncentered.append(self.AEs[j].enc1(self.shared_encoder(X)))
        return embeddings, reconstructed, uncentered
# No centers
class SharedThree(nn.Module):
    def __init__(self, n_tensors, in_feature, fst_embed, snd_embed, linear=True, CNN=False):
        # Class representing a tensorized architecture for autoencoders
        super(SharedThree, self).__init__()
        self.n_tensors = n_tensors
        self.linear = linear
        self.shared_encoder = Encoder(in_feature, fst_embed, linear)
        self.shared_decoder = Decoder(fst_embed, in_feature)
        self.AEs = nn.ModuleList()
        for i in range(n_tensors):
            self.AEs.append(Autoencoder(fst_embed, snd_embed, linear))
        
    def forward(self, X, centers, X_out=None):
        # Enters input to each tensor, and returns the embedding and the reconstruction
        batch_size=X.shape[0]
        if X_out==None:
            X_out=X
        embeddings = []
        reconstructed = []
        uncentered = []
        x = self.shared_encoder(X) 
        x_out = self.shared_encoder(X_out)
        for j in range(self.n_tensors):
            embed = self.AEs[j].enc1(x)
            decod = self.AEs[j].dec1(embed)
            embeddings.append(embed)
            reconstructed.append(self.shared_decoder(decod))
            uncentered.append(embed)
        return embeddings, reconstructed, uncentered


def train_batch_STAE(X,Y,net, n_tensors=2, lr=0.1, reg=0, epochs=100,
              number_of_batches=1, linear=True, CNN=False, X_out=None, printing=False):
    if X_out == None:
        X_out = X
    with torch.no_grad():
        centers, indices = kmeans_plusplus(X_out.numpy(), n_clusters=n_tensors, random_state=20)
        clust_assign = clust_matrix(X_out, n_tensors, centers)
        centers = clust_assign.float() @ X_out.float()
        norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X_out.shape[1],dtype=torch.float)
        centers = centers / norm    
    X = X.to(device)
    Y = Y.to(device)
    X_out = X_out.to(device)
    clust_assign = clust_assign.to(device)
    centers = centers.to(device)
    loss_f = TensorizedAEloss(in_feature=X.shape[1], reg=reg, n_tensors=n_tensors, linear=linear, CNN=CNN)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    total_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, reconstructed = net(X, centers, X_out)
        epoch_loss, assignments = loss_f(embeddings, reconstructed, X_out, centers)
        epoch_loss = epoch_loss.sum(dim=0)/X.shape[0]
        total_loss.append(epoch_loss.item())
        epoch_loss.backward()
        optimizer.step()
        # net.orthogonalize()
        for k in range(len(assignments)):
                clust_assign[:, k] = 0
                clust_assign[assignments[k]][k] = 1
        new_centers = clust_assign @ X_out
        new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1],dtype=torch.float).to(device)
        centers = new_centers / new_norm
        if printing:
            print('epoch ', epoch, ' loss ', epoch_loss)
        # elif (epoch + 1) == epochs:
        #     print('epoch ', epoch, ' loss ', epoch_loss)
        
    encoded = net.shared_encoder(X)      
    best_embeddings = torch.stack([net.AEs[assignments[i]].enc1(encoded[i]) for i in range(encoded.shape[0])], dim=0)
    encoded = torch.stack([net.shared_encoder(X[i]-centers[assignments[i]]) for i in range(X.shape[0])], dim=0)
    temp = net.shared_decoder(torch.stack([net.AEs[assignments[i]](encoded[i]) for i in range(encoded.shape[0])], dim=0))
    reconstrctions = torch.stack([temp[i]+centers[assignments[i]] for i in range(temp.shape[0])], dim=0) 
    assignments = torch.argmax(clust_assign,axis=0).numpy()
    return best_embeddings, assignments, total_loss, reconstrctions