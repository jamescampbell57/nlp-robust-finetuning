import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

class HP_BERT(Dataset):
    def __init__(self, split):
        valid_crit='run'
        fold=-1
        context_length=5

        data_dir='/home/ubuntu/nlp-robust-finetuning/brain/data/processed/'
        sub='K'
        train_runs=[1,2,3]
        test_runs=[0]
        valid_runs=[0]
        head='fmri'
        outputs='brain'

        with open(data_dir+'processed_'+sub+'.pickle', 'rb') as f:
            data = pickle.load(f)

        sentences=[]
        brain_outputs=[]

        for run in data:
            current_run_text=[]
            current_run_brain=[]
            for elem in run:
                current_run_text.append(' '.join(elem[0]))
                current_run_brain.append(torch.tensor(elem[1]))
            sentences.append(current_run_text)
            brain_outputs.append(current_run_brain)

        #sentences is [["4 words","4 words",..],[..],[..],[..]]
        #len(sentences[0]) =  305
        #len(sentences[0]) =  317
        #len(sentences[0]) =  244
        #len(sentences[0]) =  345
        #brain_outputs is [[tensor(..),tensor(..),..],[..],[..],[..]]
        #not considering window locations
        # Add context
        
        text_data = []
        for run in sentences: #_tokenized:
            text_data_run = []
            for i, _ in enumerate(run):
                #get context
                if(i>=context_length):
                    previous_sentences=run[i-context_length+1:i]
                    previous_sentences=' '.join(previous_sentences)

                else:
                    previous_sentences=run[0:i]
                    previous_sentences=' '.join(previous_sentences)

                full_sentence = ' '.join([previous_sentences, run[i]])
                text_data_run.append(full_sentence)
            text_data.append(text_data_run)

        if split=="train":
            self.dataset_text = []
            self.dataset_text.extend(text_data[1])
            self.dataset_text.extend(text_data[2])
            self.dataset_text.extend(text_data[3])
            self.dataset_brain = []
            self.dataset_brain.extend(brain_outputs[1])
            self.dataset_brain.extend(brain_outputs[2])
            self.dataset_brain.extend(brain_outputs[3])
        elif split=="test":
            self.dataset_text = text_data[0]
            self.dataset_brain = brain_outputs[0]

    def __getitem__(self, idx):
        return self.dataset_text[idx], self.dataset_brain[idx]
    
    def __len__(self):
        return len(self.dataset_text)
     
        
