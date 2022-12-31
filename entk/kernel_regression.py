import torch
import pickle
import time
import wandb
import os

root = '/home/ubuntu'
max_iter = 4500
momentum_factor = .995


def single_run(train_kernel, test_kernels, train_labels, test_labels, num_classes, lr, beta):
    train_labels_one_hot = torch.nn.functional.one_hot(train_labels, num_classes=num_classes)
    n = train_kernel.shape[0]
    alpha = torch.zeros((n, num_classes))
    
    def get_accuracy(split):
        logits = torch.matmul(test_kernels[split], alpha)
        preds = torch.argmax(logits, axis=1)
        m = test_kernels[split].shape[0]
        acc = torch.sum(preds == test_labels[split])/m
        return acc
    
    velocity = torch.zeros((n,num_classes))
    gradient = torch.ones((n,num_classes)) * .0001
    gradient_norm = -1
    iteration = 0
    while torch.any(gradient != 0) and iteration != max_iter:
        iteration += 1
        wandb.log({}, step=iteration)
        
        velocity = momentum_factor * velocity + lr * gradient
        alpha = alpha - velocity
        
        gradient = (torch.matmul(train_kernel, alpha) - train_labels_one_hot) + beta * alpha

        if iteration % 10 == 0:
            gradient_norm = torch.norm(gradient).item()
            wandb.log({"gradient norm" : gradient_norm}, step=iteration)
        if gradient_norm > 1e10:
            print("Trajectory diverged")
            iteration = max_iter
        if iteration == max_iter:
            print("Failed to reach convergence")
        elif iteration % 50 == 0:
            for test_split in test_kernels.keys():
                acc = get_accuracy(test_split)
                wandb.log({test_split : acc}, step=iteration)
            preds_train = torch.argmax(torch.matmul(train_kernel, alpha), axis=1)
            acc_train = torch.sum(preds_train == train_labels)/n
            wandb.log({"acc_train":acc_train}, step=iteration)

    return alpha



def hyperparameter_search(train_kernel, test_kernels, train_labels, test_labels, num_classes, lr_space, beta_space, dataset_name, subsample_size):
    for lr in lr_space:
        for beta in beta_space:
            run = wandb.init(project="eNTK-robustness", entity="jgc239", reinit=True)
            wandb.run.name = f"{dataset_name} lr: {lr}, beta: {beta}, momentum_factor: {momentum_factor}, subsample_size: {subsample_size}"
            wandb.config = {"learning_rate": lr, "beta": beta, "momentum_factor": momentum_factor, "subsample_size": subsample_size}

            t1 = time.time()
            
            alpha = single_run(train_kernel, test_kernels, train_labels, test_labels, num_classes, lr, beta)
            
            clock_time = time.time() - t1
            print(f"time: {clock_time}")
            run.finish()
            
            alpha_save_dir = f"{root}/eNTK-robustness/data/ntk_{dataset_name}_{subsample_size}/alphas"
            if os.path.exists(alpha_save_dir):
                torch.save(alpha, f"{alpha_save_dir}/alpha_lr_{lr}_beta_{beta}.pt")
            
'''            
if __name__ == "__main__":
    
    dataset_name = "domainnet"
    num_classes = 10 #something sketchy going on with classes
    test_splits = ["sketch_val","real_val","painting_val","clipart_val"]
    subsample_size = 500000
    
    kernel_root = f"{root}/eNTK-robustness/data/ntk_{dataset_name}_{subsample_size}/kernels"
    labels_root = f"{root}/eNTK-robustness/data/ntk_{dataset_name}_{subsample_size}/labels"
    
    train_kernel = torch.load(f"{kernel_root}/train_kernel.pt")
    test_kernels = {}
    for test_split in test_splits:
        test_kernels[test_split] = torch.load(f"{kernel_root}/{test_split}_kernel.pt")
        
    train_labels = torch.tensor(pickle.load(open(f"{labels_root}/labels_train.pkl", 'rb')))
    test_labels = {}
    for test_split in test_splits:
        test_labels[test_split] = torch.tensor(pickle.load(open(f"{labels_root}/labels_{test_split}.pkl",'rb')))

    lr_space = [1e-4]
    beta_space = [0,1e-4,10,1000]
    
    hyperparameter_search(train_kernel, test_kernels, train_labels, test_labels, num_classes, lr_space, beta_space, dataset_name, subsample_size)
'''

if __name__ == "__main__":
    dataset_name = "domainnet"
    num_classes = 40
    subsample_size = 'full'
    mega_kernel = torch.load('/home/ubuntu/eNTK-robustness/data/domainnet_mega_kernel.pt')
    train_kernel = mega_kernel[:5537,:]
    test_splits = ["sketch_val","real_val","painting_val","clipart_val"]
    test_kernels = {}
    test_kernels[test_splits[0]] = mega_kernel[5537:7936,:]
    test_kernels[test_splits[1]] = mega_kernel[7936:14879,:]
    test_kernels[test_splits[2]] = mega_kernel[14879:17788,:]
    test_kernels[test_splits[3]] = mega_kernel[17788:19404,:]
    del mega_kernel
    labels_root = f"{root}/eNTK-robustness/data/ntk_domainnet_500000/labels"
    train_labels = torch.tensor(pickle.load(open(f"{labels_root}/labels_train.pkl", 'rb')))
    test_labels = {}
    for test_split in test_splits:
        test_labels[test_split] = torch.tensor(pickle.load(open(f"{labels_root}/labels_{test_split}.pkl",'rb')))
    lr_space = [1e-7]
    beta_space = [100,10,1,.1,.01,.001,1e-4,1e-5,1e-6,1e-7]
    
    hyperparameter_search(train_kernel, test_kernels, train_labels, test_labels, num_classes, lr_space, beta_space, dataset_name, subsample_size)
    
    