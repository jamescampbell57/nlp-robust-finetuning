from utilities import select_dataset
from torch.utils.data import DataLoader
from networks import *
import quinine
from train import *

root = '/home/ubuntu/nlp-robust-finetuning/nlp'

config = quinine.Quinfig(config_path=f'{root}/configs/main_config.yaml')


def single_run(bs, lr):
    
    train_data = select_dataset(config["dataset_name"], "train", n_train=config["n_train"], n_val=config["n_val"])
    train_loader = DataLoader(train_data, batch_size=bs)

    test_loaders = []
    for test_split in config["test_splits"]:
        test_data = select_dataset(config["dataset_name"], test_split, n_train=config["n_train"], n_val=config["n_val"])
        test_loader = DataLoader(test_data, shuffle=False, batch_size=bs)
        test_loaders.append(test_loader)

    if config["model_name"] == "BERT":
        model = BERT(config["num_classes"], state_path=config["pretrained_checkpoint_path"], logits=config["logits"])
    elif config["model_name"] == "GPT2":
        model = GPT2(config["num_classes"], state_path=config["pretrained_checkpoint_path"])

    if config["algorithm"] == "linear probing":
        linear_probe(model, train_loader, test_loaders, config["test_splits"], config)
    elif config["algorithm"] == "finetuning":
        finetune(model, train_loader, test_loaders, config["test_splits"], lr, bs, config)
    elif config["algorithm"] == "LPFT":
        LPFT(model, train_loader, test_loaders, config["test_splits"], lr, bs, config["optimal_C"], config)
    elif config["algorithm"] == "FTLP":
        FTLP(model, train_loader, test_loaders, config["test_splits"], lr, bs, config)
    elif config["algorithm"] == "WiSE_FT":
        WiSE_FT(model, train_loader, test_loaders, config["test_splits"], lr, bs, config)
    elif config["algorithm"] == "LP_WiSE_FT":
        LP_WiSE_FT(model, train_loader, test_loaders, config["test_splits"], lr, bs, config["optimal_C"], config)

if config["algorithm"] == "linear probing":
    single_run(16, None)
else:   
    for bs in config["batch_size_space"]:
        for lr in config["learning_rate_space"]:
            single_run(bs,lr)
