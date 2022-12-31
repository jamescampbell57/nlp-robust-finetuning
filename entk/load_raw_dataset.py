import quinine
import utils

root = '/home/ubuntu'

def load_raw_dataset(name, split):
    ###########################################################################
    if name == "fmow":
        config_path = f'{root}/eNTK-robustness/configs/adaptation/fmow_all.yaml'
        split_to_idx = {
            "id_val" : 0,
            "ood_val" : 1,
            "test" : 2,
            "africa_test" : 3
        }
    elif name == "living17":
        config_path = f'{root}/eNTK-robustness/configs/adaptation/living17.yaml'
        split_to_idx = {
            "source" : 0,
            "target" : 1
        }
    elif name == "ent30":
        config_path = f'{root}/eNTK-robustness/configs/adaptation/entity30.yaml'
        split_to_idx = {
            "source" : 0,
            "target" : 1
        }
    elif name == "domainnet":
        config_path = f'{root}/eNTK-robustness/configs/adaptation/domainnet.yaml'
        split_to_idx = {
            "sketch_val" : 0,
            "real_val" : 1,
            "painting_val" : 2,
            "clipart_val" : 3
        }
    elif name == "cifarstl":
        config_path = f'{root}/eNTK-robustness/configs/adaptation/cifar_stl.yaml'
        split_to_idx = {
            "cifar_test" : 0,
            "stl_test" : 1
        }      
    #############################################################################
    quinfig = quinine.Quinfig(config_path=config_path)
    if split == "train":
        dataset = utils.init_dataset(quinfig['train_dataset'])
    else:
        test_config = quinfig['test_datasets'][split_to_idx[split]]
        if 'transforms' not in test_config:
            test_config['transforms'] = quinfig['default_test_transforms']
        dataset = utils.init_dataset(test_config)
    return dataset