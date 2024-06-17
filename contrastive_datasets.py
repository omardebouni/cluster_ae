import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_circles, make_moons
from torchvision import datasets, transforms
import random

# Function to sample points on a sphere
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)  # Generate random points
    vec /= np.linalg.norm(vec, axis=0)    # Normalize to lie on a sphere
    return vec

# Function to create multiple sets of concentric circles with noise and class separation
def make_many_circles(num_samples, noise_level, class_labels_list):
    # Initialize empty lists to hold the features (X) and labels (Y)
    X_list = []
    Y_list = []
    n_classes = len(class_labels_list)    # Number of classes
    num_samples = num_samples * 2         # Adjust sample size for inner and outer circles

    for class_label in range(n_classes):
        # Generate a set of circles
        X_full, Y_full = make_circles(n_samples=num_samples // n_classes, noise=noise_level, factor=0.5)
        
        # Select only the outer circle (label=0 in Y_full)
        idx = Y_full == 0
        X_selected = X_full[idx]
        Y_selected = np.full(X_selected.shape[0], class_label)  # Assign class label
        
        # Adjust positions to separate the classes visually
        shift_x, shift_y = np.random.rand(2) * 5 - 1  # Random shift
        X_selected[:, 0] += shift_x * class_label
        X_selected[:, 1] += shift_y * class_label
        
        # Append to the lists
        X_list.append(X_selected)
        Y_list.append(Y_selected)
    
    # Concatenate all pieces
    X_full = np.vstack(X_list)
    Y_full = np.concatenate(Y_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(np.arange(X_full.shape[0]))
    X = torch.tensor(X_full[shuffle_idx], dtype=torch.float)
    Y = torch.tensor(Y_full[shuffle_idx])
    
    return X, Y

# Class to generate a dataset with only positive samples
class OnlyPosi(Dataset):
    def __init__(self, dataset, num_triplets=1, n_pos=1, isImg=False, noise_level=0.01):
        self.dataset = dataset
        self.num_triplets = num_triplets
        self.n_pos = n_pos
        self.isImg = isImg
        self.noise_level = noise_level
        
    def __len__(self):
        return len(self.dataset) * self.num_triplets
    
    def __getitem__(self, idx):
        index = idx // self.num_triplets
        img, label = self.dataset[index]
        
        # Generate positive samples
        positives = []
        for _ in range(self.n_pos):
            noise = torch.randn(img.size()) * self.noise_level
            pos = img + noise
            if self.isImg:
                pos = torch.clamp(pos, 0.0, 1.0)  # Ensure the image values are within valid range
            positives.append(pos)
        positives = torch.stack(positives)        
        return img, positives, index
        
# Class to generate a dataset with both positive and negative samples
class Many(Dataset):
    def __init__(self, dataset, num_triplets=1, n_pos=1, n_neg=-1, isImg=False, noise_level=0.01, contra=True):
        self.dataset = dataset
        self.num_triplets = num_triplets
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.isImg = isImg
        self.noise_level = noise_level
        self.contra = contra
        
    def __len__(self):
        return len(self.dataset) * self.num_triplets
    
    def __getitem__(self, idx):
        index = idx // self.num_triplets
        img, label = self.dataset[index]
        
        # Generate positive samples
        positives = []
        for _ in range(self.n_pos):
            noise = torch.randn(img.size()) * self.noise_level
            pos = img + noise
            if self.isImg:
                pos = torch.clamp(pos, 0.0, 1.0)  # Ensure the image values are within valid range
            positives.append(pos)
            
        # Generate negative samples if contrastive learning is enabled
        if self.contra:
            negatives = []
            negative_indices = []  # Keep track of the indices of the negatives
            while len(negatives) < self.n_neg:
                neg_idx = np.random.randint(0, len(self.dataset))
                if neg_idx != index:  # Ensure negative sample is different from the anchor
                    neg, _ = self.dataset[neg_idx]
                    negatives.append(neg)
                    negative_indices.append(neg_idx)
            negatives = torch.stack(negatives)
            
        positives = torch.stack(positives)
        
        if self.contra: 
            return img, positives, negatives, index
        else:
            return img, positives, index

# Function to create a data loader for contrastive learning datasets
def contrastive_data_loader(generator='many', dataset='mnist', contra=True, class_labels_list=[0, 1], num_samples=1, num_aug=1, normalise_data=True, isImg=True, noise_level=0.1, n_pos=1, n_neg=1):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'mnist':
        # Load the MNIST dataset
        data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print('MNIST loaded dataset', data)
        X_full = data.data.view(data.data.size(0), -1).float() / 255  # Normalize the data
        Y_full = data.targets
        x_idx = torch.tensor([]).long()
        
        # Select only items from class_labels_list
        for i in class_labels_list:
            x_idx = torch.cat((x_idx, (Y_full == i).nonzero(as_tuple=True)[0]))
            
        # Randomize and cut indices
        shuffled = torch.randperm(x_idx.size(0))[:num_samples]
        indices = x_idx[shuffled]
        X = X_full[indices]
        Y = Y_full[indices]
    elif dataset == 'moons':
        # Load the moons dataset
        X_full, Y_full = make_moons(n_samples=num_samples, noise=0)
        x_idx = torch.tensor(np.arange(X_full.shape[0]))
        shuffle_idx = torch.randperm(x_idx.shape[0])
        X = torch.tensor(X_full[shuffle_idx], dtype=torch.float)
        Y = torch.tensor(Y_full[shuffle_idx])
    elif dataset == 'circles':
        # Load the circles dataset
        X, Y = make_many_circles(num_samples, noise_level, class_labels_list)
    elif dataset == 'sphere':
        # Generate sphere data
        sphere_data = sample_spherical(num_samples, ndim=3).T
        sphere_labels = np.zeros(num_samples)
    
        # Generate ring data
        r = np.random.uniform(2, 2.4, size=(num_samples,))
        angle = np.linspace(0, 2 * np.pi, num_samples)
        ring_data = np.array([r * np.cos(angle), r * np.sin(angle), np.zeros(num_samples)]).T
        ring_labels = np.ones(num_samples)
    
        # Combine data and labels
        X_full = np.concatenate((sphere_data, ring_data), axis=0)
        Y_full = np.concatenate((sphere_labels, ring_labels), axis=0)
        # Shuffle and subsample
        indices = [i for i in range(len(X_full))]
        indices = random.sample(indices, len(indices))[:num_samples]
        X = X_full[indices]
        Y = Y_full[indices]
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.float).squeeze()
    else:
        raise Exception("Not implemented")
    
    # Create the appropriate dataset generator
    if generator == 'many':  
        return Many(list(zip(X, Y)), num_triplets=num_aug, isImg=isImg, noise_level=noise_level, n_pos=n_pos, n_neg=n_neg, contra=contra), X, Y
    else:
        return OnlyPosi(list(zip(X, Y)), num_triplets=num_aug, isImg=isImg, noise_level=noise_level, n_pos=n_pos), X, Y
