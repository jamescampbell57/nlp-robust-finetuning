import torch
from tqdm import tqdm
import pickle
import os
from load_raw_dataset import load_raw_dataset

root = "/home/ubuntu"

def compute_eNTK(model, dataset_name, split, subsample_size=500000, seed=123):
    
    dataset = load_raw_dataset(dataset_name, split)
    
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = list(model.parameters())
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    save_dir = f"{root}/robust-finetuning/data/ntk_{dataset_name}_{subsample_size}"
    if not os.path.exists(save_dir):
        os.system(f"mkdir {save_dir}")
    if not os.path.exists(f"{save_dir}/{split}"):
        os.system(f"mkdir {save_dir}/{split}")
    
    random_index = torch.randperm(num_params)[:subsample_size]
    for i in tqdm(range(len(dataset))):
        model.zero_grad()
        model.forward(torch.unsqueeze(dataset[i][0], 0).to(device))[0].backward()
        eNTK = []
        for param in params:
            if param.requires_grad:
                eNTK.append(param.grad.flatten())
        eNTK = torch.cat(eNTK)
        #subsampling
        ntk_data_point = torch.clone(eNTK[random_index])
        torch.save(ntk_data_point, f"{save_dir}/{split}/ntk_{i}.pt")
        
    labels_dir = f"{root}/robust-finetuning/data/ntk_{dataset_name}_{subsample_size}/labels"
    labels_file = f"{labels_dir}/labels_{split}.pkl"
    store_labels(dataset, labels_dir, labels_file)
      
def store_labels(raw_dataset, save_dir, save_file):
    labels = []
    for i in tqdm(range(len(raw_dataset))):
        labels.append(raw_dataset[i][1])
    if not os.path.exists(save_dir):
        os.system(f"mkdir {save_dir}")
    if not os.path.exists(save_file):
        os.system(f"touch {save_file}")
    pickle.dump(labels, open(save_file, 'wb'))
                  
if __name__ == "__main__":
    #add parser
    from construct_model import build_model
    import quinine
    config_path = f"{root}/eNTK-robustness/configs/adaptation/cifar_stl.yaml"
    config = quinine.Quinfig(config_path)
    model = build_model(config)
    print("Starting to compute NTK")
    for split in ["train", "cifar_test", "stl_test"]:
        compute_eNTK(model, "cifarstl", split)
    
    