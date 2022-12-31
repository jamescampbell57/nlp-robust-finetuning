import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from ntk_datasets import NTK_Dataset

from torch.optim import Adam
from tqdm import tqdm
import wandb


class OverparameterizedLinearRegression(nn.Module):
    def __init__(self, num_classes, subsample_size=500000):
        super().__init__()
        self.linear = nn.Linear(subsample_size, num_classes)
        
    def forward(self, x):
        return self.linear(x)
    
    
def train(model, train_loader, eval_loaders, eval_names, num_classes, learning_rate, num_epochs=10, beta=0):
    run = wandb.init(project="eNTK-robustness", entity="jgc239", reinit=True)
    wandb.run.name = f"ent30 lr: {learning_rate}, beta: {beta}"
    wandb.config = {"learning_rate": learning_rate, "num_epochs": num_epochs, "beta": beta}
    lr = learning_rate
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss() #need to implement regularization
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    progress_bar = tqdm(range(num_epochs*len(train_loader)))
    
    model.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            #need to synchronize wandb steps
            x = x.to(device)
            preds = model(x)
            y = nn.functional.one_hot(y, num_classes=num_classes)
            y = y.to(device).to(torch.float)
            loss = loss_function(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"training loss": loss.item()})
            progress_bar.update(1)
            if i % 1000 == 0:
                for eval_name, eval_loader in zip(eval_names, eval_loaders):
                    score = evaluate(model, eval_loader)
                    wandb.log({eval_name : score})
                    lr = lr/2
                    optimizer = Adam(model.parameters(), lr=lr)
    run.finish()

def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    for x, y in dataloader:
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, axis=1)
            y = y.to(device).to(torch.int64)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)
    return float(num_correct)/float(num_samples)*100
    

if __name__ == "__main__":
    
    batch_size = 16
    learning_rate = 1e-2
    num_classes = 30
    
    train_data = NTK_Dataset("ent30", "train")
    random_index = torch.randperm(6000)[:500]
    id_val_data = Subset(NTK_Dataset("ent30", "source"), random_index)
    ood_val_data = Subset(NTK_Dataset("ent30", "target"), random_index)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    id_val_loader = DataLoader(id_val_data, batch_size=batch_size, shuffle=False)
    ood_val_loader = DataLoader(ood_val_data, batch_size=batch_size, shuffle=False)
    
    eval_loaders = [id_val_loader, ood_val_loader]
    eval_names = ['id_val','ood_val']
    
    regressor = OverparameterizedLinearRegression(num_classes)
    
    train(regressor, train_loader, eval_loaders, eval_names, num_classes, learning_rate, num_epochs=10, beta=0)
    
    