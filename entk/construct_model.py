import torch
import numpy as np
import utils


def build_model(config):
    net = utils.initialize(config['model'])
    # If fine-tune, re-initialize the last layer.
    finetune = 'finetune' in config and config['finetune']
    linear_probe = 'linear_probe' in config and config['linear_probe']
    freeze_bottom_k = 'freeze_bottom_k' in config
    batch_norm = 'batchnorm_ft' in config and config['batchnorm_ft']
    side_tune = 'side_tune' in config and config['side_tune']
    def count_parameters(model, trainable):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)
    if finetune or linear_probe or batch_norm or side_tune:
        if freeze_bottom_k:
            # Currently only implemented for some models (including CLIP ViTs).
            net.freeze_bottom_k(config['freeze_bottom_k']) 
        if linear_probe:
            print('linear probing, freezing bottom layers.')
            # If unspecified, we set use_net_val_mode = True for linear-probing.
            # We did this in update_net_eval_mode which we called in main.
            assert('use_net_val_mode' in config)
            # Freeze all the existing weights of the neural network.
            # TODO: enable keeeping the top linear layer.
            net.set_requires_grad(False)
        if batch_norm:
            assert(not linear_probe)
            print("tuning only batch norm layers and lin probe")
            net.set_requires_grad(False)
            for layer in net._model.modules():
                if isinstance(layer, nn.modules.batchnorm.BatchNorm2d): 
                    for param in layer.parameters():
                        param.requires_grad = True 
        if 'probe_net' in config:
            probe_net = utils.initialize(config['probe_net'])
            net.add_probe(probe_net)
        else:
            net.new_last_layer(config['num_classes'])
        if side_tune:
            # This is currently only supported for some networks like ResNet-50,
            # would need to add support for other networks.
            net.enable_side_tuning()
        if ('linear_probe_checkpoint_path' in config and
            config['linear_probe_checkpoint_path'] != ''):
            linprobe_path = config['linear_probe_checkpoint_path']
            coef, intercept, best_c, best_i = pickle.load(open(linprobe_path, "rb"))
            if coef.shape[0] == 1:
                # For binary classification, sklearn returns a 1-d weight
                # vector. So we convert it into a 2D vector with the same
                # logits. To see this conversion, notice that if I have a
                # binary weight vector w, the output is \sigma(w^T x)
                # = e^(w^T x) / (1 + e^(w^T x)). On the other hand,
                # if I have weight vector [w/2, -w/2], and I use softmax
                # I get e^(w^T x / 2) / (e^(w^T x / 2) + e^(-w^T x / 2)
                # and multiplying num / denom by e^(w^T x / 2)
                # we get the same expression as above.
                coef = np.concatenate((-coef/2, coef/2), axis=0)
                intercept = np.array([-intercept[0]/2, intercept[0]/2])
            if 'normalize_lp' in config and config['normalize_lp']:
                print("Normalizing linear probe std-dev")
                saved_stddev = np.std(coef)
                rand_weights = net.get_last_layer().weight.detach().numpy()
                rand_stddev = np.std(rand_weights)
                print(
                    "Weights saved stddev: %f, desired stddev: %f",
                    saved_stddev, rand_stddev)
                print("Intercept saved stddev: %f", np.std(intercept))
                coef = (coef / saved_stddev) * rand_stddev
                intercept = (intercept / saved_stddev) * rand_stddev
                print(
                    "Final stddev weights; %f, intercept: %f",
                    np.std(coef), np.std(intercept))
                # What should it be, based on rep size (maybe get from net last layer)
                # Divide
            net.set_last_layer(coef, intercept)
        if 'isolate_grads' in config:
            net.isolate_grads()
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        print(f'Fine Tuning {num_trainable_params} of {num_params} parameters.')
    if 'checkpoint_path' in config and len(config['checkpoint_path']) > 0:
        print(utils.load_ckp(config['checkpoint_path'], net))
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        print(f'Fine Tuning checkpoint: {num_trainable_params} of {num_params} parameters.')
    return net
