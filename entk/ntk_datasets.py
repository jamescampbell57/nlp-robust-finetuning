import torch
from torch.utils.data import Dataset
import pickle

root = '/home/ubuntu'

class NTK_Dataset(Dataset):
    def __init__(self, name, split, subsample_size=500000):
        self.data_path = f"{root}/eNTK-robustness/data/ntk_{name}_{subsample_size}/{split}"
        labels_path = f"{root}/eNTK-robustness/data/ntk_{name}_{subsample_size}/labels/labels_{split}.pkl"
        self.labels = pickle.load(open(labels_path,'rb'))
    def __getitem__(self, idx):
        x = torch.load(f"{self.data_path}/ntk_{idx}.pt")
        y = self.labels[idx]
        return x, y
    def __len__(self):
        return len(self.labels)