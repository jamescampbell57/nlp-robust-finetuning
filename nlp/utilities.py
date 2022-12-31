from dataset_classes import *

def select_dataset(dataset_name, split, **kwargs):
    if dataset_name == "Amazon":
        return Amazon(split, **kwargs)
    elif dataset_name == "MNLI_1":
        return MNLI(split, 1, **kwargs)
    elif dataset_name == "MNLI_2":
        return MNLI(split, 2, **kwargs)
    elif dataset_name == "SST2_IMDB":
        return SST2_IMDB(split, **kwargs)
    elif dataset_name == "STSB":
        return STSB(split, **kwargs)
    elif dataset_name == "Yelp":
        return Yelp(split, **kwargs)
    elif dataset_name == "HarryPotter":
        return HP_BERT(split)
    raise ValueError("Dataset could not be selected")

#def fine_tune(model, method):
#    if method == "fine-tune":
#        training_methods.fine_tune(model)


def change_all_keys(pre_odict):
    def change_key(odict, old, new):
        for _ in range(len(odict)):
            k, v = odict.popitem(False)
            odict[new if old == k else k] = v
            return odict
    for key in pre_odict.keys():
        if key[:5] == 'bert.':
            post_odict = change_key(pre_odict, key, key[5:])
            return change_all_keys(post_odict)
        if key[:7] == 'linear.':
            del pre_odict[key]
            return change_all_keys(pre_odict)
    return pre_odict
        