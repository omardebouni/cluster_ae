import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.cluster import adjusted_rand_score
from models import TensorisedAEloss
from new_models import PartSharedLoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_AE(net, X, Y, lr=0.1, epochs=100, CNN=False, X_out=None):
    X_out = X_out if X_out is not None else X
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.1)
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            x, x_out = X[i], X_out[i]
            if CNN:
                x = x.reshape(28, 28).unsqueeze(0).unsqueeze(0)
                x_out = x_out.reshape(28, 28).unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            out = net(x.float())
            loss = criterion(out, x_out.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / X.shape[0]
        train_loss.append(epoch_loss)

    return train_loss


def train_TAE(X, Y, n_clusters=2, lr=0.1, reg=0, embed=2, epochs=100, number_of_batches=1, linear=True, CNN=False, shared=(False, None, None), X_out=None, printing=False, sig=False):
    X_out = X_out if X_out is not None else X
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    centers, indices = kmeans_plusplus(X_out.cpu().detach().numpy(), n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(X_out - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float64).to(device)
    for i in range(X.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ X_out.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
    centers = centers / norm
    _, fst_embed, scnd_embed = shared

    if shared[0]:
        net = PartSharedLoss(X.shape[1], fst_embed, scnd_embed, reg=reg, num_clusters=n_clusters, linear=linear, CNN=CNN, sig=sig).to(device)
    else: 
        net = TensorisedAEloss(X.shape[1], fst_embed=fst_embed, scnd_embed=scnd_embed, reg=reg, num_clusters=n_clusters, linear=linear, CNN=CNN).to(device)
        
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    for epoch in range(epochs):
        total_loss = 0
        batch_size = int(X.shape[0] / number_of_batches)
        
        for b in range(int(X.shape[0] / batch_size)):
            optimizer.zero_grad()
            temp_idx = []
            batch_loss = 0

            for i in range(batch_size):
                j = b * batch_size + i
                loss_sample, idx = net(X[j].float(), centers, j, clust_assign, X_out[j].float())
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss /= batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ X_out.float()
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
            new_centers /= new_norm
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / X.shape[0]
        if printing:
            print('epoch', epoch, 'loss', epoch_loss)
            print("ARI:", adjusted_rand_score(torch.argmax(clust_assign, axis=0).cpu().detach().numpy(), Y.cpu().detach().numpy()))
        elif (epoch + 1) == epochs:
            print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return net, train_loss, clust_assign, X, Y, X_out


def train_CTAE(X, Y, n_clusters=2, lr=0.1, reg=0, embed=2, epochs=100, number_of_batches=1, linear=True, CNN=False, X_out=None):
    X_out = X_out if X_out is not None else X
    X, Y, X_out = X.to(device), Y.to(device), X_out.to(device)
    train_loss = []

    centers, indices = kmeans_plusplus(X_out.cpu().detach().numpy(), n_clusters=n_clusters, random_state=20)
    clust = torch.cat([torch.norm(X_out - torch.tensor(center).to(device), dim=1).reshape(-1, 1) for center in centers], dim=1)
    clust = torch.argmin(clust, axis=1).to(device)
    clust_assign = torch.zeros([n_clusters, X.shape[0]], dtype=torch.float64).to(device)
    for i in range(X_out.shape[0]):
        clust_assign[clust[i], i] = 1

    centers = clust_assign.float() @ X_out.float()
    norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X_out.shape[1], dtype=torch.float).to(device)
    centers /= norm

    net = TensorisedCAEloss(X_out.shape[1], embed, num_clusters=n_clusters).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        batch_size = int(X.shape[0] / number_of_batches)
        
        for b in range(number_of_batches):
            optimizer.zero_grad()
            temp_idx = []
            batch_loss = 0

            for i in range(batch_size):
                j = b * batch_size + i
                loss_sample, idx = net(X[j].float(), centers, j, clust_assign, X_out[j].float())
                batch_loss += loss_sample
                temp_idx.append(idx)
                total_loss += loss_sample.item()

            batch_loss /= batch_size
            batch_loss.backward(retain_graph=True)
            optimizer.step()

            for k in range(batch_size):
                kb = b * batch_size + k
                clust_assign[:, kb] = 0
                clust_assign[temp_idx[k]][kb] = 1

            new_centers = clust_assign.float() @ X_out.float()
            new_norm = torch.sum(clust_assign, axis=1, dtype=torch.float).reshape(-1, 1) @ torch.ones(1, X.shape[1], dtype=torch.float).to(device)
            new_centers /= new_norm
            centers = (b * centers + new_centers) / (b + 1)

        epoch_loss = total_loss / X.shape[0]
        print('epoch', epoch, 'loss', epoch_loss)
        train_loss.append(epoch_loss)

    return net, train_loss, clust_assign, X, Y, X_out
