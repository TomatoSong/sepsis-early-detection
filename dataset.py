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
import sys
import math

from utils import get_patient_by_id_original, get_patient_by_id_standardized, get_patient_data, get_synthetic_patient_by_id
from config import All_COLS, COL_IDX_MAP

DATA_FOLDER = "../data"
processing = {
    "original"    : "/",
    "imputed"     : "/imputed3/",
    "normalized"  : "/normalized3/",
    "standardized": "/standardized/",
    "standardized_padded": "/standardized_padded/"
}

class SepsisDataset(Dataset):
    def __init__(self, patient_ids, config, method='standardized'):
        self.seq_len = config["window_size"]
        self.horizon = config["horizon"]
        self.columns = config["columns"]
        self.start_offset = config["start_offset"]
        
        self.col_indices = [COL_IDX_MAP[i] for i in self.columns]
        self.col_indices.sort()
        
        self.patient_ids = patient_ids
        self.method = method
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(self.seq_len, self.start_offset)
        
    def check_store(self, seq_len, start_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'_(\d+)'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == start_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            self.ratio = info['ratio']
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, start_offset):
        index_map_subset = self.check_store(seq_len, start_offset)
        if not index_map_subset:
            path = '../data/idxmap_'+str(seq_len)+'_'+str(start_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_standardized(pid)
                    if not (p['SepsisLabel'] == 0).all():
                        sepsis_6 = p['SepsisLabel'].idxmax()
                    else:
                        sepsis_6 = -1
                    if len(p) < start_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        hypoxia = int(p.at[t, 'Hypoxia'])
                        u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
                        hist = (pid,          # patient id
                                0,            # start
                                t,            # end
                                label,        # sepsis label
                                u_weights,    # utility weights
                                seq_len-t-1,  # padding
                                sepsis_6,     # sepsis timestamp (-6)
                                hypoxia       # hypoxia label
                               )
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[5] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(start_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            hypoxia = int(p.at[t, 'Hypoxia'])
                            u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
                            if t < seq_len:
                                hist = (pid,          # patient id
                                        0,            # start
                                        t,            # end
                                        label,        # sepsis label
                                        u_weights,    # utility weights
                                        seq_len-t-1,  # padding
                                        sepsis_6,     # sepsis timestamp (-6)
                                        hypoxia       # hypoxia label
                                       )  
                            else: 
                                hist = (pid,          # patient id
                                        t-seq_len+1,  # start
                                        t,            # end
                                        label,        # sepsis label
                                        u_weights,    # utility weights
                                        0,            # padding
                                        sepsis_6,     # sepsis timestamp (-6)
                                        hypoxia       # hypoxia label
                                       )
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
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
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
            dirname = '../data/idxmap_subset/'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            path = dirname + 'idxmap_subset_'+str(seq_len)+'_'+str(start_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids,
                               ratio = self.ratio,
                               index_map_subset=index_map_subset
                              ), 
                          fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, u_weights, padding, sepsis_6, hypoxia = self.idxmap_subset[idx]
        data = get_patient_data(pid, start, end, self.col_indices)
        data = [[0]*len(self.columns)]*padding + data
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        if self.horizon != 6 and sepsis_6 > 0:
            sepsis_horizon = sepsis_6 - (self.horizon - 6)
            if sepsis_horizon <= end:
                label = 1
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, mask, empty_tensor
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(u_weights), torch.tensor(mask), torch.tensor([])


class MultiTaskDataset(SepsisDataset):
    def __init__(self, patient_ids, config, method='standardized'):
        self.seq_len = config["window_size"]
        self.horizon = config["horizon"]
        self.columns = config["columns"]
        self.start_offset = config["start_offset"]
        self.lead = config["lead"]
        self.forecast_len = config["forecast_len"]
        self.target = config["target"]
        
        self.col_indices = [COL_IDX_MAP[i] for i in self.columns]
        self.col_indices.sort()

        self.patient_ids = patient_ids
        self.method = method
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map()

    def check_store(self):
        try:
            directory = '../data/mt_idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'_(\d+)'
                matches = re.findall(pattern, filename)
                if len(matches) < 6:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if (int(matches[0]) == self.seq_len and
                    int(matches[1]) == self.start_offset and
                    int(matches[2]) == self.horizon and
                    int(matches[3]) == self.lead and
                    int(matches[4]) == self.forecast_len):
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            self.ratio = info['ratio']
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self):
        seq_len = self.seq_len
        start_offset = self.start_offset
        index_map_subset = self.check_store()
        
        if not index_map_subset:
            path = '../data/mt_{}_{}_{}_{}_{}.json'.format(
                self.seq_len,
                self.start_offset,
                self.horizon,
                self.lead,
                self.forecast_len
            )
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(40336), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_standardized(pid)

                    if len(p) < self.start_offset + self.forecast_len + self.lead:
                        continue

                    if not (p['SepsisLabel'] == 0).all():
                        sepsis_time = p['SepsisLabel'].idxmax() + 6
                    else:
                        sepsis_time = math.inf

                    for t in range(start_offset-1,len(p)-self.forecast_len):
                        predict_time = t + self.horizon
                        sepsis_label = int(predict_time >= sepsis_time)
                        hypoxia = int(p.at[predict_time, 'Hypoxia']) if predict_time < len(p) else int(p['Hypoxia'].iloc[-1])
                        u_weights = [p.loc[t, 'UtilityNeg'], p.loc[t, 'UtilityPos']]
                        forecast_start = t + 1 + self.lead
                        forecast_end   = forecast_start + self.forecast_len - 1
                        assert forecast_end < len(p)
                        assert forecast_end - forecast_start + 1 == self.forecast_len
                        start = 0 if t < seq_len else t-seq_len+1
                        padding = seq_len-t-1 if t < seq_len else 0
                        assert t - start + 1 + padding == seq_len
                        hist = (pid,              # patient id
                                start,            # start
                                t,                # end
                                padding,          # padding
                                sepsis_label,     # sepsis label with the current horizon
                                sepsis_time,      # sepsis timestamp
                                u_weights,        # utility weights
                                forecast_start,   # forecast start
                                forecast_end,     # forecast end
                                hypoxia           # hypoxia label with the current horizon
                               )  
                        assert len(hist) == 10
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            label = sepsis_label if self.target == 'Sepsis' else hypoxia
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                print('populated {} patients into {} timeseries'.format(len(patients_all), len(index_map)))
                with open(path, "w") as fp:
                    json.dump(index_map, fp)
            else:
                with open(path, "r") as fp:
                    index_map = json.load(fp)
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
                        pid = item[0]
                        label = item[4] if self.target == 'Sepsis' else item[9] 
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))
            print('len idxmap {}'.format(len(index_map)))
            dirname = '../data/mt_idxmap_subset/'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            path = dirname + 'idxmap_subset_{}_{}_{}_{}_{}_{}.json'.format(
                self.seq_len,
                self.start_offset,
                self.horizon,
                self.lead,
                self.forecast_len,
                len(index_map_subset)
            )
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids,
                               ratio = self.ratio,
                               index_map_subset=index_map_subset
                              ), 
                          fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset

    def __getitem__(self, idx):
        pid, start, end, padding, sepsis, _, u, f_start, f_end, hypoxia = self.idxmap_subset[idx]
        data = get_patient_data(pid, start, end, self.col_indices)
        data = [[0]*len(self.columns)]*padding + data
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        future = get_patient_data(pid, f_start, f_end, self.col_indices)
        label = sepsis if self.target == 'Sepsis' else hypoxia 
        
        ### Return:
        # patient_id
        # last_hour
        # clinical_data
        # label
        # utility_weights
        # mask
        # forcast clinical_data
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(u), torch.tensor(mask), torch.tensor(future)                      


class RawDataset(Dataset):
    def __init__(self, pids, config):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        self.x = []
        self.y = []
        self.u = []
        self.ids = []
        for pid in tqdm(pids, desc="Preparing raw data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            self.x.extend(p[All_COLS].values.tolist())
            self.y.extend(p['SepsisLabel'].tolist())
            u = p.apply(lambda row: [row['UtilityNeg'], row['UtilityPos']], axis=1).tolist()
            self.u.extend(u)
            self.ids.extend([(pid, rid) for rid in range(len(p))])
        print('Populated {} dps from {} patients'.format(len(self.y), len(pids)))
        return
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, empty_tensor, empty_tensor
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.u[idx], dtype=torch.float32), torch.tensor([]), torch.tensor([])


class SyntheticDataset(Dataset):
    def __init__(self, pids, seq_len, start_offset, columns):
        self.seq_len = seq_len
        self.patient_ids = pids
        self.columns = columns
        self.ratio = [0,0]
        self.idxmap_subset = self.build_index_map(seq_len, start_offset)

    def check_store(self, seq_len, start_offset):
        try:
            directory = '../data/idxmap_subset/'
            for filename in os.listdir(directory):
                pattern = r'syn_[^0-9]*(_\d+)+'
                matches = re.findall(pattern, filename)
                if len(matches) < 2:
                    print("Illegal idx map file found {}!".format(filename))
                    continue
                if int(matches[0]) == seq_len and int(matches[1]) == start_offset:
                    f = os.path.join(directory, filename)
                    with open(f, "r") as fp:
                        info = json.load(fp)
                        if info['patient_ids'] == self.patient_ids:
                            return info['index_map_subset']
        except Exception as e:
            print(f"An error occurred during file search: {e}")
        return None
        
    def build_index_map(self, seq_len, start_offset):
        index_map_subset = self.check_store(seq_len, start_offset)
        if not index_map_subset:
            path = '../data/syn_idxmap_'+str(seq_len)+'_'+str(start_offset)+".json"
            index_map = []
            index_map_subset = []
            patients_subset = set()
            patients_all = set()
            if not os.path.exists(path):
                for pid in tqdm(range(10000), desc="Building idx map", ascii=False, ncols=75):
                    p = get_patient_by_id_original(pid)
                    if len(p) < start_offset:
                        t = len(p)-1
                        label = int(p.at[t, 'SepsisLabel'])
                        hist = (pid,0,t,label,seq_len-t-1) # patient id, start, end, label, padding
                        assert hist[2] < len(p)
                        assert hist[2]-hist[1]+1+hist[4] == seq_len
                        index_map.append(hist)
                        patients_all.add(pid)
                        if pid in self.patient_ids:
                            index_map_subset.append(hist)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
                    else:
                        for t in range(start_offset-1,len(p)):
                            label = int(p.at[t, 'SepsisLabel'])
                            hist = (pid,0,t,label, seq_len-t-1) if t < seq_len else (pid,t-seq_len+1,t,label,0)
                            assert hist[2] < len(p)
                            assert hist[2]-hist[1]+1+hist[4] == seq_len
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
                    for item in tqdm(index_map, desc="Building idx map subset", ascii=False, ncols=75):
                        pid = item[0]
                        label = item[-1]
                        if pid in self.patient_ids:
                            index_map_subset.append(item)
                            self.ratio[label] += 1
                            patients_subset.add(pid)
            print('using {} patients in current subset'.format(len(patients_subset)))

            assert len(patients_subset) == len(self.patient_ids)
            assert patients_subset == set(self.patient_ids)
            print('len idxmap {}'.format(len(index_map)))
            path = '../data/idxmap_subset/syn_idxmap_subset_'+str(seq_len)+'_'+str(start_offset)+'_'+str(len(index_map_subset))+".json"
            with open(path, "w") as fp:
                json.dump(dict(patient_ids=self.patient_ids, index_map_subset=index_map_subset), fp)
        
        print('len idxmap subset {}'.format(len(index_map_subset)))
        return index_map_subset
    
    def get_ratio(self):
        return self.ratio
        
    def __len__(self):
        return (len(self.idxmap_subset))

    def __getitem__(self, idx):
        pid, start, end, label, padding = self.idxmap_subset[idx]
        data = [0]*padding + get_patient_data(pid, start, end)
        mask = [True]*padding + [False]*(self.seq_len-padding)
        assert len(data) == self.seq_len
        assert len(mask) == self.seq_len
        return pid, end, torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), torch.tensor(mask), torch.tensor([])


class WeibullCoxDataset(Dataset):
    def __init__(self, pids):
        self.build_index_map(pids)
        
    def build_index_map(self, pids):
        path = '../data/wc_dataset.json'
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if set(data['pids']) == set(pids):
                    self.x = data['x']
                    self.tau = data['tau']
                    self.S = data['S']
                    self.ids = data['ids']
                    self.y = data['y']
                    self.u = data['u']
                    return
        except Exception as err:
            print('Failed matching saved Weibull-Cox dataset: ', err)
        
        self.x = []
        self.tau = []
        self.S = []
        self.ids = []
        self.y = []
        self.u = []
        for pid in tqdm(pids, desc="Preparing data", ascii=False, ncols=75):
            p = get_patient_by_id_standardized(pid)
            for rid in range(len(p)-5):
                window = p.iloc[rid:min(len(p),rid+6),:]
                S = 0 if (window['SepsisLabel'] == 1).any() else 1
                tau = max(0.1, window['SepsisLabel'].idxmax()-rid if (window['SepsisLabel'] == 1).any() else 7)
                x = p.loc[rid, All_COLS].astype(float).tolist()
                y = p.loc[rid, ['SepsisLabel']].astype(int).tolist()
                u = [p.loc[rid, 'UtilityNeg'], p.loc[rid, 'UtilityPos']]
                self.x.append(x)
                self.y.append(y)
                self.u.append(u)
                self.tau.append(tau)
                self.S.append(S)
                self.ids.append((pid, rid))

        data = {
            'pids': pids,
            'x'   : self.x,
            'tau' : self.tau,
            'S'   : self.S,
            'y'   : self.y,
            'u'   : self.u,
            'ids' : self.ids
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return
        
    def __len__(self):
        return len(self.S)
    
    def __getitem__(self, idx):
        # Return: patient_id, latest_hour, clinical_data, label, utility_weights, tau, S
        return self.ids[idx][0], self.ids[idx][1], torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), torch.tensor(self.u[idx], dtype=torch.float32), torch.tensor(self.tau[idx], dtype=torch.float32), torch.tensor(self.S[idx], dtype=torch.int32)