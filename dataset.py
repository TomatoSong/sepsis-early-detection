import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np

DATA_FOLDER = "../data"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3/",
    "normalized"  : "/normalized3/",
    "standardized": "/standardized/"
}

path1 = "../data/training/"
path2 = "../data/training_setB/"
fnames1 = os.listdir(path1)
fnames2 = os.listdir(path2)
fnames1.sort()
fnames2.sort()

COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
       'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
       'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
       'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'ICULOS']

class SepsisDataset(Dataset):
    def __init__(self, seq_len=72, starting_offset=24, cols=COLS, method='standardized'):
        self.folder_path = DATA_FOLDER + processing[method]
        self.cols = cols
        self.method = method
        self.seq_len = seq_len
        self.idxmap = self.build_index_map(seq_len, starting_offset)
        
    def build_index_map(self, seq_len, starting_offset):
        path = '../data/idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
        if not os.path.exists(path):
            index_map = []
            for idx in tqdm(range(40336), desc="Builind idx map", ascii=False, ncols=75):
                p = get_patient_by_id_original(idx)
                sepsis = 1 in p.SepsisLabel.unique()
                pred_hour = p.SepsisLabel.ne(0).idxmax()
                for t in range(starting_offset-1,len(p)+1):
                    label = 1 if (sepsis and t >= pred_hour) else 0
                    hist = (idx,0,t,label) if t < seq_len else (idx,t-seq_len+1,t,label)
                    index_map.append(hist)
            with open(path, "w") as fp:
                json.dump(index_map, fp)
        else:
            with open(path, "r") as fp:
                index_map = json.load(fp)
        return index_map
    
    def pad_sequence_and_create_mask(self, seq):
        offset = self.seq_len-len(seq)
        new_rows = pd.DataFrame(0, index=np.arange(offset), columns=seq.columns, dtype='float32')
        padded_sequence = pd.concat([new_rows, seq])
        attention_mask = [True if i < offset else False for i in range(self.seq_len)]
        return padded_sequence.values.tolist(), attention_mask
        
    def __len__(self):
        return len(self.idxmap)

    def __getitem__(self, idx):
        pid, start, end, label = self.idxmap[idx]
        p = pd.read_csv(self.folder_path+'p'+str(pid).zfill(6)+'.csv')
        p = p[self.cols]
        seq = p.iloc[start:end]
        padded_seq, mask = self.pad_sequence_and_create_mask(seq)
        return Tensor(padded_seq), Tensor(mask), label
