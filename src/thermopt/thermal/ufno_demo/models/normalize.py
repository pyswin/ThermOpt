import torch
import torch.nn as nn
from torch.utils.data import Dataset


class normalize(nn.Module):
    def __init__(self, x0, if_trainable=False):
        super().__init__()
        self.mean = nn.Parameter(x0.mean(), requires_grad=if_trainable)
        self.std = nn.Parameter(x0.std(), requires_grad=if_trainable)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class normalize_3d:
    def __init__(self, data, if_trainable=False):
        super().__init__()
        self.min_val = nn.Parameter(torch.min(data), requires_grad=if_trainable)
        self.max_val = nn.Parameter(torch.max(data), requires_grad=if_trainable)

    def forward(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse(self, scaled_data):
        return scaled_data * (self.max_val - self.min_val) + self.min_val


def cal_rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


class CombinedDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = len(x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]
