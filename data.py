import numpy as np
import torch
import torchvision.datasets as datasets
from sklearn.datasets import load_iris
import pandas as pd

def sample_data(mu_list, sigma_list, dummy_dim, n, seed=0, noise=0):
    """
    n:      number of datapoints
    mu:     torch array means for each cluster
    sigma:
    dummy_dim: number of dim to add std gaussian noise
    seed:   set random seed for sampling

    create dataset of Gaussian clusters with intrinsic dim derived from mu dim
    and actual dim is mu dim + dummy_dim
    """
    np.random.seed(seed)

    c_size = len(mu_list)
    X = np.zeros((n, len(mu_list[0]) + dummy_dim))
    Y = np.zeros((n))

    for i in range(c_size):
        C_i = np.random.multivariate_normal(mu_list[i], sigma_list[i], int(n / c_size))
        Y_i = np.ones((int(n / c_size))) * i

        dummy = np.random.normal(0, 0.1, int(n / c_size) * dummy_dim)
        dummy = dummy.reshape(int(n / c_size), -1)

        c = np.concatenate((C_i, dummy), axis=1)

        X[int(n / c_size) * i:int(n / c_size) * (i + 1), :] = c
        Y[int(n / c_size) * i:int(n / c_size) * (i + 1)] = Y_i

    if noise != 0:
        add_noise = noise * np.random.normal(0, 1, X.shape[0] * X.shape[1])
        add_noise = add_noise.reshape(X.shape[0], X.shape[1])
        X_noise = X.copy() + add_noise

        return X, Y, X_noise

    return X, Y, None


def real_data_loader(dataset='mnist', class_labels_list=[0, 1, 2, 3, 4], num_samples=50, normalise_data=False,
                     subsample=True, CNN=False):
    """
    penguin4:  dataset='penguin4', class_labels_list=[0,1,2], normalise_data=True, subsample=False
    iris:     dataset='iris', class_labels_list=[0,1,2], normalise_data=False, subsample=False
    mnist:    dataset='mnist', class_labels_list=[0,1,2,3,4], num_samples=200, normalise_data=False, subsample=True
    fashion:  dataset='fashion', class_labels_list=[0,1,2,3,4], num_samples=200, normalise_data=False, subsample=True
    cifar10:  dataset='cifar10', class_labels_list=[0,1,2,3,4], num_samples=200, normalise_data=False, subsample=True
    """

    if dataset == 'cifar10':
        data = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        print('CIFAR10 loaded dataset', data)

        X_full = data.data.reshape(data.data.shape[0], -1)
        Y_full = data.targets

    if dataset == 'mnist':
        data = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        print('MNIST loaded dataset', data)
        if CNN:
            # Keep the original 28x28 shape for CNN
            X_full = data.data.unsqueeze(1)  # Add a channel dimension
        else:
            X_full = data.data.reshape(data.data.shape[0], -1)
        Y_full = data.targets

    if dataset == 'fashion':
        data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
        print('Fashion MNIST loaded dataset', data)
        if CNN:
            # Keep the original 28x28 shape for CNN
            X_full = data.data.unsqueeze(1)  # Add a channel dimension
        else:
            X_full = data.data.reshape(data.data.shape[0], -1)
        Y_full = data.targets

    if dataset == 'caltech':
        data = datasets.Caltech101(root='./data', download=True, transform=None)
        print('Caltech loaded dataset', data)

    if (dataset == 'penguin4') or (dataset == 'penguin2'):
        peng = pd.read_csv("data/penguins_size.csv")
        peng = peng.dropna()

        if dataset == 'penguin2':
            X_full = np.array(peng[['culmen_length_mm', 'culmen_depth_mm']])
        else:
            X_full = np.array(peng[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']])

        if normalise_data:
            x_norm = np.max(X_full, axis=0)
            # print("NORM ", x_norm)
            X_full = X_full[:, :] / x_norm
            

        # change the labels
        peng_dict = {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}
        peng = peng.replace({'species': peng_dict})
        Y_full = np.array(peng['species'])
        # print(Y_full)

    if dataset == 'iris':
        iris = load_iris()
        X_full, Y_full = iris.data, iris.target

    if subsample:
        # pick classes in class_labels_list, num_samples from each
        x_idx = torch.tensor([])
        for i in class_labels_list:
            x_idx = torch.cat((x_idx, (Y_full == i).nonzero(as_tuple=True)[0][:num_samples]))
        print('Number of total samples ', x_idx.shape)
    else:
        x_idx = torch.tensor(np.arange(X_full.shape[0]))
#     if normalise_data:
#             x_norm = np.max(X_full, axis=0)
#             print("NORM ", x_norm)
#             X_full = X_full[:, :] / x_norm

    shuffle_idx = torch.randperm(x_idx.shape[0])
    x_idx = x_idx[shuffle_idx].long()
    X = torch.tensor(X_full[x_idx], dtype=torch.float)
    Y = torch.tensor(Y_full[x_idx])

    return X, Y

def add_noise_data(X, noise=0.1, clip=False):
    add_noise = noise*np.random.normal(0,1,X.shape[0]*X.shape[1])
    add_noise = add_noise.reshape(X.shape[0],X.shape[1])
    X_noise = X.clone() + add_noise
    if clip:
        X_noise = np.clip(X_noise,0.,1.)
    return X_noise.float()

def parallel_line(noise=0.1):
    mu_list = np.array([[-1, -1], [4, -2]]) + np.random.normal(0,0,(2,2))
    sigma_list = np.array([[[5, 4.95],[4.95, 5]],
              [[5, 4.95],[4.95, 5]]
             ])+ np.random.normal(0,0,(2,2,2))
    X,Y,X_noise = sample_data(mu_list, sigma_list, dummy_dim=3, n = 150, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    if noise>0:
        X_noise = torch.tensor(X_noise, dtype=torch.float32)
    return X,Y,X_noise,2

def orthogonal(noise=0.1):
    mu_list = [[-1, -1], [7, 7]]+ np.random.normal(0,0,(2,2))
    sigma_list = np.array([[[5, 4.9],[4.9, 5]],  [[5, -4.9],[-4.9, 5]]])+ np.random.normal(0,0,(2,2,2))
    X,Y,X_noise = sample_data(mu_list, sigma_list, dummy_dim=3, n = 150, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    if noise>0:
        X_noise = torch.tensor(X_noise, dtype=torch.float32)
    return X,Y,X_noise,2

def triangle(noise=0.1):
    mu_list = [[-1, -1], [6, 6],[5,-5]] + np.random.normal(0,0,(3,2))
    sigma_list = [[[5, 4.9],[4.9, 5]],  [[5, -4.5],[-4.5, 5]],[[7, 0],[0, 1]]] + np.random.normal(0,0,(3,2,2))

    X,Y,X_noise = sample_data(mu_list, sigma_list, dummy_dim=3, n = 150, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    if noise>0:
        X_noise = torch.tensor(X_noise, dtype=torch.float32)
    return X,Y,X_noise,3

def lines_3D(noise=0.1):
    mu_list = [[2, 0, 0], [0, 0, 4], [2,2,3]]
    sigma_list = [[[3, 0.01, 0.01],[0.01, 0.01, 0.01],[0.01, 0.01, 0.01]],
              [[0.01, 0.01, 0.01],[0.01, 3, 0.01],[0.01, 0.01, 0.01]],
              [[0.01,0.01,0.01],[0.01,0.01,0.01],[0.01,0.01,3]]
             ]

    X,Y,X_noise = sample_data(mu_list, sigma_list, dummy_dim=3, n = 300, noise=noise)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    if noise>0:
        X_noise = torch.tensor(X_noise, dtype=torch.float32)
    return X,Y,X_noise,3

