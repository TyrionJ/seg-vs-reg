import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import pkload


class TrainingDataset(Dataset):
    def __init__(self, data_path, atlas_file, transforms):
        self.paths = data_path
        self.atlas_file = atlas_file
        self.transforms = transforms

    @staticmethod
    def one_hot(img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        x, x_seg = pkload(self.paths[index])
        y, y_seg = pkload(self.atlas_file)

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class ValidationDataset(Dataset):
    def __init__(self, data_path, atlas_file, transforms):
        self.atlas_file = atlas_file
        self.paths = data_path
        self.transforms = transforms

    @staticmethod
    def one_hot(img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        x, x_seg = pkload(self.atlas_file)
        y, y_seg = pkload(self.paths[index])

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x_seg, y_seg = torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
