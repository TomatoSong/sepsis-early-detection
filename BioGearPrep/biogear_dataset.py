import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import re
import ast

from BioGearPrep.utils_biogear import get_patient_data_biogear, get_patient_by_id_standardized_biogear

DATA_FOLDER = "../data/BioGearData/"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3_biogear/",
    "normalized"  : "/normalized3_bigoear/",
    "standardized": "/standardized_biogear/",
    "standardized_padded": "/standardized_padded_biogear/"
}

COLS = ['patient_id', 'time', 'BaseExcess', 'BUN', 'CTSresistance', 'HR', 'MAP', 'SAP', 'DAP', 'CO', 'HgbContent', 'CVP', 'Hct', 'ABpH', 'UR', 'WBC', 'UPR', 'Resp', 'O2Sat', 'CO2Sat', 'Ctemp', 'Stemp',
        'Bilirubin_total', 'Phosphate', 'HCO3', 'Creatinine', 'Lactate', 'Calcium', 'Chloride', 'Glucose', 'Potassium', 'Hgb', 'PaCO2', 'qsofa', 'sepsis']

class BioGearDataset(Dataset):
    def __init__(self, pids, seq_len=72, starting_offset=24, cols=COLS, method='standardized'):
        self.patient_ids = pids
        self.method = method
        self.seq_len = seq_len
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, starting_offset)
    
    def check_store(self, seq_len, starting_offset):
        try:
            directory = '../data/idxmap_subset_biogear/'
            for filename in os.listdir(directory):
                pattern = r'_(\d+)'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == starting_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
    
    def build_index_map(self, seq_len, starting_offset):
        index_map_subset = self.check_store(seq_len, starting_offset)
        if not index_map_subset:
            path = '../data/biogear_idxmap_'+str(seq_len)+'_'+str(starting_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(190), desc="Building idx map", ascii=False, ncols=36):
                    p = get_patient_by_id_standardized_biogear(pid)
                    #print(p["UtilityPos"])
                    if len(p) < starting_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'sepsis'])
                        u_weights = ast.literal_eval(p.at[t, 'UtilityScores'])
                        hist = (pid,0,t,label,u_weights,seq_len-t-1) # patient id, start, end, label, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[5] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(starting_offset-1,len(p)):
                            label = int(p.at[t, 'sepsis'])
                            u_weights = ast.literal_eval(p.at[t, 'UtilityScores'])
                            hist = (pid,0,t,label,u_weights,seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,u_weights,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[5] == seq_len
                            index_map.append(hist)
                            patients_all.add(pid)
                            if pid in self.patient_ids:
                                index_map_subset.append(hist)
                                self.ratio[label] += 1
                                patients_subset.add(pid)
                print('populated {} patients into {} timeseries'.format(len(patients_all), len(index_map)))
                with open(path, "w") as fp:
                    json.dump(index_map, fp)
            else:
                with open(path, "r") as fp:
                    index_map = json.load(fp)
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=36):
                        pid = item[0]
                        label = item[3]
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))

            assert len(patients_subset) == len(self.patient_ids)
            assert patients_subset == set(self.patient_ids)
            print('len idxmap {}'.format(len(index_map)))
            path = '../data/idxmap_subset_biogear/idxmap_subset_biogear_'+str(seq_len)+'_'+str(starting_offset)+'_'+str(len(index_map_subset))+".json"
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids, index_map_subset=index_map_subset), fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset

    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, u_weights, padding = self.idxmap_subset[idx]
        data = [0]*padding + get_patient_data_biogear(pid, start, end)
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, mask, empty_tensor
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(u_weights), torch.tensor(mask), torch.tensor([])