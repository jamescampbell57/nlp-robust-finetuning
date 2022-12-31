import torch
from torch.utils.data import DataLoader
import os
from ntk_datasets import NTK_Dataset
from tqdm import tqdm

root = '/home/ubuntu'
subsample_size = 500000
chunk_size = 1000

def get_train_kernel(name):
    train_dataset = NTK_Dataset(name, "train")
    loader = DataLoader(train_dataset, batch_size=chunk_size, shuffle=False)
    n = len(train_dataset)
    train_kernel = torch.zeros((n,n))

    for i, batch_i in tqdm(enumerate(loader), desc=' outer loop', position=0):
        for j, batch_j in tqdm(enumerate(loader), desc=' inner loop', position=1, leave=False):
            if i <= j:
                patch = torch.matmul(batch_i[0], batch_j[0].T)

                start_i = i*chunk_size
                end_i = (i+1)*chunk_size if (i+1)!=len(loader) else n
                start_j = j*chunk_size
                end_j = (j+1)*chunk_size if (j+1)!=len(loader) else n

                train_kernel[start_i:end_i,start_j:end_j] = patch

                if i != j:
                    train_kernel[start_j:end_j,start_i:end_i] = patch.T
                
    save_path = f"{root}/eNTK-robustness/data/ntk_{name}_{subsample_size}/kernels"
    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")
    if not os.path.exists(f"{save_path}/train_kernel.pt"):
        os.system(f"touch {save_path}/train_kernel.pt")
    torch.save(train_kernel, f"{save_path}/train_kernel.pt")

def get_test_kernel(name, split):
    train_dataset = NTK_Dataset(name, "train")
    train_loader = DataLoader(train_dataset, batch_size=chunk_size, shuffle=False)
    n = len(train_dataset)
    test_dataset = NTK_Dataset(name, split)
    test_loader = DataLoader(test_dataset, batch_size=chunk_size, shuffle=False)
    n_prime = len(test_dataset)
    
    test_kernel = torch.zeros((n_prime, n))
    
    for i, batch_i in tqdm(enumerate(train_loader), desc=' outer loop', position=0):
        for j, batch_j in tqdm(enumerate(test_loader), desc=' inner loop', position=1, leave=False):
            patch = torch.matmul(batch_j[0], batch_i[0].T)
            
            start_i = i*chunk_size
            end_i = (i+1)*chunk_size if (i+1)!=len(train_loader) else n
            start_j = j*chunk_size
            end_j = (j+1)*chunk_size if (j+1)!=len(test_loader) else n_prime
            
            test_kernel[start_j:end_j,start_i:end_i] = patch
            
    save_path = f"{root}/eNTK-robustness/data/ntk_{name}_{subsample_size}/kernels"
    if not os.path.exists(save_path):
        os.system(f"mkdir {save_path}")
    if not os.path.exists(f"{save_path}/{split}_kernel.pt"):
        os.system(f"touch {save_path}/{split}_kernel.pt")
    torch.save(test_kernel, f"{save_path}/{split}_kernel.pt")
    
    
if __name__ == "__main__":
    dataset_name = "cifarstl"
    test_splits = ["cifar_test", "stl_test"]
    get_train_kernel(dataset_name)
    for test_split in test_splits:
        get_test_kernel(dataset_name, test_split)
