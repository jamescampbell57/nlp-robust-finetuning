import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import BertModel, GPT2ForSequenceClassification
from utilities import change_all_keys

class BERT(nn.Module):
    def __init__(self, num_out, return_cls_representation=False, state_path=None, to_device=True, seed=42, logits=False):
        super().__init__()
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        if state_path is not None:
            pre_odict = torch.load(state_path)['model_state_dict']
            filtered_odict = change_all_keys(pre_odict)
            self.bert.load_state_dict(filtered_odict, strict=True)
        self.linear = nn.Linear(768,num_out)
        self.return_cls_representation = return_cls_representation
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to_device = to_device
        self.logits = logits
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        if self.to_device:
            embeddings = embeddings.to(self.device)
        representations = self.bert(**embeddings).last_hidden_state
        cls_representation = representations[:,0,:]
        if self.return_cls_representation:
            return cls_representation
        pred = self.linear(cls_representation)
        if self.logits:
            return pred
        return self.softmax(pred)


class GPT2(nn.Module):
    def __init__(self, num_out, state_path=None, to_device=True, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = 50259
        if state_path is not None: #LOOK INTO THIS, THIS IS SKETCHY
            self.gpt = GPT2ForSequenceClassification.from_pretrained(state_path, num_labels=num_out, ignore_mismatched_sizes=True)
        else:
            self.gpt = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_out)
        self.gpt.config.pad_token_id = 50259
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to_device = to_device
    def forward(self, x):
        embeddings = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True) #NO PADDING OR TRUNCATION, SUPER SUS
        if self.to_device:
            embeddings = embeddings.to(self.device)
        return self.gpt(**embeddings).logits
           
        
        
        
