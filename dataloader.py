import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pickle


# This file is adapted from the PENN GitHub project (https://github.com/xiaolong-snnu/PENN).
# Special thanks to the original authors for providing the foundational code.

class MyDataset(Dataset):
    def __init__(self, file_name, max_sample=-1):
        """
        Initialize the dataset by loading data from a file and padding trajectories to the same length.

        Args:
            file_name (str): Path to the dataset file.
            max_sample (int): Maximum number of samples to load (for debugging).
        """
        super(MyDataset, self).__init__()
        print('Loading data from', file_name)

        # Load data from pickle file
        with open(file_name, "rb") as fp:
            data_dic, param_dic = pickle.load(fp)

        X = data_dic['X']
        X = [torch.from_numpy(x) for x in X]
        Y = data_dic['Y']
        H = data_dic['H']
        self.param_dic = param_dic

        # If max_sample is set, limit the dataset size for debugging purposes
        if max_sample > 0:
            print('*' * 50)
            print('max_sample =', max_sample)
            print('*' * 50)
            X = X[:max_sample]
            Y = Y[:max_sample]
            H = H[:max_sample]

        # Record the lengths of each trajectory and find the maximum length
        self.lengths = [_.shape[0] for _ in X]
        self.max_lengths = max(self.lengths)

        # Pad all trajectories to the maximum length
        # Use ZeroPad2d to add zero padding and ensure all trajectories have the same length
        self.X = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_lengths - _.shape[0]))(_) for _ in X])
        self.Y = torch.from_numpy(np.array(Y))
        self.H = torch.from_numpy(np.array(H).reshape((-1, 1)))
        self.Z = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_lengths - i))(torch.ones((i, 1))) for i in self.lengths])
        self.lengths = torch.from_numpy(np.reshape(np.array(self.lengths), (-1, 1))).float()

        print('=' * 50)
        print('Dataset summary:')
        print('X size:', self.X.shape)  # Trajectories
        print('Y size:', self.Y.shape)  # Parameters
        print('H size:', self.H.shape)  # Spanning times
        print('Z size:', self.Z.shape)  # Zero masks
        print('=' * 50)

    def get_dim(self):
        """
        Get the input and output dimensions for the model.

        Returns:
            tuple: Input dimension and output dimension.
        """
        return self.X.shape[1], self.Y.shape[1]

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Trajectory (x), parameters (y), length (l), spanning time (h), and zero mask (z).
        """
        x = self.X[index, :]  # The trajectories
        y = self.Y[index, :]  # The parameters
        l = self.lengths[index, :]  # Lengths of trajectories
        h = self.H[index, :]  # Spanning times
        z = self.Z[index, :]  # The zero masks for the trajectories with varying lengths
        return x, y, l, h, z

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.X)
