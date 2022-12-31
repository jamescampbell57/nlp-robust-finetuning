import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
import wandb
import numpy as np
from sklearn.linear_model import LogisticRegression




#AdamW, MSELoss not configurable
def train(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, num_epochs, dataset_name, num_classes, track_train_acc=True, weights_save_path=None, num_iterations_per_save=99999999999):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-6, amsgrad=True)
    #####
    #checkpoint = torch.load('/home/ubuntu/nlp-robust-finetuning/data/bert_checkpoints/checkpoint_13')
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #optimizer.to('cuda')
    #####
    loss_function = torch.nn.MSELoss()
    steps_per_epoch = len(train_loader)
    num_training_steps = num_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=.1*num_training_steps, num_training_steps=num_training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for iteration, batch in enumerate(train_loader):
            global_iteration=steps_per_epoch*epoch+iteration
            preds = model(list(batch[0]))
            targets = F.one_hot((batch[1]).to(torch.int64), num_classes=num_classes).to(device)
            loss = loss_function(preds, targets.float())
            #loss = loss_function(preds, batch[1].float().to(device)) I think for fMRI pretraining, logit matching
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            wandb.log({"training loss": loss.item()}, step=global_iteration)
            progress_bar.update(1)
            if weights_save_path is not None and iteration % num_iterations_per_save == 0 and epoch == 0: #only use for first epoch
                save_checkpoint(weights_save_path, dataset_name, learning_rate, batch_size, epoch, loss, model, optimizer, iteration=global_iteration)
        if track_train_acc:
            train_score = evaluate(model, train_loader, num_classes)
            wandb.log({"train acc": train_score}, step=global_iteration)
        for test_split, test_loader in zip(test_splits, test_loaders):
            #NO CHEAP EVALUATE
            val_score = evaluate(model, test_loader, num_classes)
            wandb.log({test_split: val_score}, step=global_iteration)
        if weights_save_path is not None:
            save_checkpoint(weights_save_path, dataset_name, learning_rate, batch_size, epoch, loss, model, optimizer)
            
def save_checkpoint(weights_save_path, dataset_name, learning_rate, batch_size, epoch, loss, model, optimizer, iteration='last'):
    if not os.path.exists(weights_save_path):
        os.system(f'mkdir {weights_save_path}')
    if not os.path.exists(f'{weights_save_path}/{dataset_name}'):
        os.system(f'mkdir {weights_save_path}/{dataset_name}')
    if not os.path.exists(f'{weights_save_path}/{dataset_name}/bs_{batch_size}_lr_{learning_rate}'):
        os.system(f'mkdir {weights_save_path}/{dataset_name}/bs_{batch_size}_lr_{learning_rate}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iteration': iteration
        }, f'{weights_save_path}/{dataset_name}/bs_{batch_size}_lr_{learning_rate}/checkpoint_{epoch}_{iteration}') #save path name of epoch


def cheap_evaluate(model, dataloader, num_classes):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    loss_function = torch.nn.MSELoss()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(list(batch[0]))
            losses.append(torch.norm(preds-batch[1].float().to(device)))
    return torch.tensor(losses).sum()/300 #VERY HEURISTIC
    
def evaluate(model, dataloader, num_classes):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            preds = model(list(batch[0]))
            preds = torch.argmax(preds, axis=1)
            labels = F.one_hot((batch[1]).to(torch.int64), num_classes=num_classes).to(device)
            labels = torch.argmax(labels, axis=1)
            num_correct += (preds==labels).sum()
            num_samples += preds.size(0)
    model.train()
    return float(num_correct)/float(num_samples)*100 


def setup_wandb(dataset_name, n_train, n_val, algorithm, learning_rate, num_epochs, batch_size, model_name, config, optional_description='', lin_probe_epoch='N/A', lin_probe_iter='N/A'):
    wandb_config = {
            "dataset": dataset_name,
            "n_train": n_train,
            "n_val": n_val,
            "model": model_name,
            "algorithm": algorithm,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "optional_description": optional_description,
            "FTLP epoch": lin_probe_epoch,
            "FTLP iter": lin_probe_iter
        }
    run_name = f"{dataset_name} {model_name} {algorithm} lr: {learning_rate}, bs: {batch_size} "+optional_description
    if lin_probe_epoch != 'N/A':
        run_name = run_name+' '+str(lin_probe_epoch)+' '+str(lin_probe_iter)
    run = wandb.init(project=config["wandb_project"], entity=config["wandb_username"], reinit=True, config=wandb_config, name=run_name)
    return run


def finetune(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config):
    run = setup_wandb(config["dataset_name"], config["n_train"], config["n_val"], config["algorithm"], learning_rate, config["num_epochs"], batch_size, config["model_name"], config, optional_description=config["optional_description"])
    train(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config["num_epochs"], config["dataset_name"], config["num_classes"], weights_save_path=config["weights_save_path"], num_iterations_per_save=config["num_iterations_per_save"])
    run.finish()
    

def LPFT(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, optimal_C, config):
    assert optimal_C != None, "optimal_C is none"
    run = setup_wandb(config["dataset_name"], config["n_train"], config["n_val"], "LPFT", learning_rate, config["num_epochs"], batch_size, config["model_name"], config, optional_description=config["optional_description"])
    X_train, y_train, X_test_list, y_test_list = get_last_layer_representations(model, train_loader, test_loaders)
    
    classifier = LogisticRegression(random_state=0, warm_start=True, max_iter=config["max_iter"], C=optimal_C)
    classifier.fit(X_train, y_train)
    
    sk_weight = torch.tensor(classifier.coef_).float()
    sk_bias = torch.tensor(classifier.intercept_).float()
    sk_weight = torch.nn.Parameter(data=sk_weight)
    sk_bias = torch.nn.Parameter(data=sk_bias)
    
    model.linear.weight = sk_weight
    model.linear.bias = sk_bias
    train(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config["num_epochs"], config["dataset_name"], config["num_classes"])
    run.finish()
    
    
def linear_probe(model, train_loader, test_loaders, test_splits, config, **kwargs):
    if len(kwargs) != 0:
        run = setup_wandb(config["dataset_name"], config["n_train"], config["n_val"], config["algorithm"], kwargs['FTLP_lr'], kwargs['FTLP_epoch'], kwargs['FTLP_bs'], config["model_name"], config, optional_description=config["optional_description"], lin_probe_epoch=kwargs['FTLP_epoch'], lin_probe_iter=kwargs['FTLP_iter'])
    else:
        run = setup_wandb(config["dataset_name"], config["n_train"], config["n_val"], config["algorithm"], "N/A", "N/A", "N/A", config["model_name"], config, optional_description=config["optional_description"], lin_probe_epoch="N/A", lin_probe_iter="N/A")
    
    X_train, y_train, X_test_list, y_test_list = get_last_layer_representations(model, train_loader, test_loaders)
    max_iter = config["max_iter"]
    classifier = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iter)
    start_C = config["start_C"]
    end_C = config["end_C"]
    num_Cs = config["num_Cs"]
    Cs = np.logspace(start_C, end_C, num_Cs)
    all_scores = {}
    for C in Cs:
        classifier.C = C
        classifier.fit(X_train, y_train)
        scores_for_given_C = {}
        #selection assumes in-distribution validation is first in test_splits
        for X_test, y_test, test_split in zip(X_test_list, y_test_list, test_splits):
            new_score = classifier.score(X_test, y_test)*100
            wandb.log({test_split : new_score, "C value": C})
            scores_for_given_C[test_split] = new_score
        all_scores[C] = scores_for_given_C
    optimal_C, results = find_optimal_C(all_scores, test_splits)
    results["optimal_C"] = optimal_C
    wandb.log(results)
    run.finish()
    #can return for iterated linearization
    
def find_optimal_C(scores, test_splits):
    highest = 0
    corresponding_C = 0
    best_inner_dict = {}
    for C, inner_dict in scores:
        id_val_score_for_given_C = inner_dict[test_splits[0]]
        if id_val_score_for_given_C >= highest:
            highest = id_val_score_for_given_C
            corresponding_C = C
            best_inner_dict = inner_dict
    return corresponding_C, best_inner_dict
    
def get_last_layer_representations(model, train_loader, test_loaders):
    #may not be true outside of custom BERT
    model.to(model.device)
    model.return_cls_representation = True
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_loader)):
            batch_representations = model(list(batch[0]))
            X_train = batch_representations if i ==0 else torch.cat((X_train, batch_representations))
            y_train = batch[1] if i==0 else torch.cat((y_train, batch[1]))
        X_train = X_train.to('cpu').numpy()
        y_train = y_train.to('cpu').numpy()

        X_test_list = []
        y_test_list = []
        for test_loader in test_loaders:
            for i, batch in tqdm(enumerate(test_loader)):
                batch_representations = model(list(batch[0]))
                X_test = batch_representations if i ==0 else torch.cat((X_test, batch_representations))
                y_test = batch[1] if i==0 else torch.cat((y_test, batch[1]))
            X_test_list.append(X_test.to('cpu').numpy())
            y_test_list.append(y_test.to('cpu').numpy())
    model.return_cls_representation = False
    model.train()
    #returns numpy arrays on the cpu
    return X_train, y_train, X_test_list, y_test_list


    
#note: interpolating between final linear weights and given initialization
def WiSE_FT(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config):      
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    theta_0 = model.state_dict()
    checkpoints_dir = os.path.join(config["weights_save_path"], config["dataset_name"], f'bs_{batch_size}_lr_{learning_rate}')
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_{config["wise_epoch"]}_{config["wise_iteration"]}')
    theta_1 = torch.load(checkpoint_path)["model_state_dict"]
    assert set(theta_0.keys()) == set(theta_1.keys())
    
    run = setup_wandb(config["dataset_name"], config["n_train"], config["n_val"], config["algorithm"], learning_rate, config["num_epochs"], batch_size, config["model_name"], config, optional_description=config["optional_description"])
    
    alphas = np.linspace(config["start_alpha"], config["end_alpha"], config["num_alphas"])
    for iteration, alpha in enumerate(alphas):
        theta = {key: (1-alpha)*theta_0[key] + alpha*theta_1[key] for key in theta_0.keys()}
        model.load_state_dict(theta)
        wandb.log({"alpha": alpha}, step=iteration)
        for test_split, test_loader in zip(test_splits, test_loaders):
            val_score = evaluate(model, test_loader, config["num_classes"])
            wandb.log({test_split: val_score}, step=iteration)
    #can call find_optimal_C to automatically return optimal alpha
    run.finish()
        
#note: this uses random head for initial conditions of training, then fits head after the fact. Different from ideal LP_WiSE_FT       
def LP_WiSE_FT(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, optimal_C, config):
    assert optimal_C != None, "optimal_C is none"
    X_train, y_train, X_test_list, y_test_list = get_last_layer_representations(model, train_loader, test_loaders)
    classifier = LogisticRegression(random_state=0, warm_start=True, max_iter=config["max_iter"], C=optimal_C)
    classifier.fit(X_train, y_train)
    sk_weight = torch.tensor(classifier.coef_).float()
    sk_bias = torch.tensor(classifier.intercept_).float()
    sk_weight = torch.nn.Parameter(data=sk_weight)
    sk_bias = torch.nn.Parameter(data=sk_bias)
    model.linear.weight = sk_weight
    model.linear.bias = sk_bias
    WiSE_FT(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config)

    
    
def initialize_head_with_LP():    
    #may make sense to isolate function that takes in a model with random init head and return model with LP head
    return None

def WiSEr_FT(): 
    '''
    #2 implementations of WiSE_FT for non-zero-shot models
    #first do LPFT, then interpolate using LP'd weights; in other words, make it into a zero shot model, then do WiSE-FT
    #alternate implementation LP -> WiSE-FT
    #w_0 -> w_T1
    #theta_0, w_T1 -> theta_T2, w_T2
    #
    #theta_F = alpha*theta_0 + (1-alpha)*theta_T2
    #w_F = alpha*w_T1 + (1-alpha)*w_T2
    #
    #
    #second algorithm: don't interpolate linear head, then retrain
    #theta_0, w_0 -> theta_T, w_T
    #theta_F1 = alpha*theta_0 + (alpha-1)*theta_T
    #w_F1 = w_T
    #re-train to adjust mismatch between features being used and features learned (can also FT-LP from here)
    #theta_F1, w_F1 -> theta_F2, w_F2'''
    #first implementation
    
    return None

#def fit_linear_head(model, train_loader, test_loaders, test_splits, config):
#    X_train, y_train, X_test_list, y_test_list = get_last_layer_representations(model, train_loader, test_loaders)
#    max_iter = config["max_iter"]
#    classifier = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iter)
#    start_C = config["start_C"]
#    end_C = config["end_C"]
#    num_Cs = config["num_Cs"]
#    Cs = np.logspace(start_C, end_C, num_Cs)
#    for C in Cs:
#        classifier.C = C
#        classifier.fit(X_train, y_train)
#        for X_test, y_test, test_split in zip(X_test_list, y_test_list, test_splits):
#            wandb.log({test_split : classifier.score(X_test, y_test), "C value": C})
    
#strategy: first run train and save model checkpoints, then loop linear_probe through saved checkpoints
def FTLP(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config):
    if config["obtain_checkpoints"]:
        finetune(model, train_loader, test_loaders, test_splits, learning_rate, batch_size, config)
    checkpoints_dir = os.path.join(config["weights_save_path"], config["dataset_name"], f'bs_{batch_size}_lr_{learning_rate}')
    sorted_checkpoints = sorted((f for f in os.listdir(checkpoints_dir) if not f.startswith(".")), key=str.lower)
    for checkpoint_name in sorted_checkpoints:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        #loss = checkpoint['loss']
        FTLP_settings = {'FTLP_lr' : learning_rate,
                         'FTLP_bs' : batch_size,
                         'FTLP_epoch' : epoch,
                         'FTLP_iter' : iteration} 
        linear_probe(model, train_loader, test_loaders, test_splits, config, **FTLP_settings)
    
