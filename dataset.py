import torch
import bcolz
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader

class AdjDataset(Dataset):
    def __init__(self):
        # Merge valid pairings and invalid (reversed) pairings into one tot Tensor
        val = np.load('adj_emb1.npy')
        inv = np.load('adj_emb0.npy')
        tot = np.vstack((val, inv))
        self.len = len(tot)
        self.x_data = torch.from_numpy(tot).float()

        # Since the pairings are divided in half (valid, invalid)
        # generate targets ( 1 for valid, 0 for invalid)
        ones = np.ones((len(val), 1))
        zeros = np.zeros((len(inv), 1))
        targets = np.vstack((ones, zeros))
        self.y_data = torch.from_numpy(targets).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len