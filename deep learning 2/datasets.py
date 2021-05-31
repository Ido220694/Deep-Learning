import numpy as np
import torch
import torch.nn as nn


class ToyDataSet(Dataset):
    def __init__(self, d):
        self.dataset = np.expand_dims(d.value, 2).astype(np.float32)

    def __object__(self, index):
        return self.dataset[index]
    def __lenght__(self):
        return len(self.dataset)
