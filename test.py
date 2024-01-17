import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

from dataset import SepsisDataset
from transformer import TimeSeriesClassifier
from config import *
from utils import get_patient_by_id_original

def test(test_ids, model, rid):
    
    test_dataset = SepsisDataset(test_ids, seq_len=SEQ_LEN, starting_offset=SEQ_OFFSET)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    model.eval()

    results = {}
    with open("../results/"+rid+".json", "r") as f:
        results = json.load(f)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pids, t_ends, sequences, masks, labels = batch
            masks = masks.bool()
            sequences, masks = sequences.to(device), masks.to(device)
            outputs = model(sequences, masks)
            probabilities = torch.sigmoid(outputs)
            predictions = outputs.argmax(dim=1)

            for pid, timestamp, predicted_prob, predicted_label, true_label in zip(pids.tolist()
                                                                                 , t_ends.tolist()
                                                                                 , probabilities.tolist()
                                                                                 , predictions.tolist()
                                                                                 , labels.tolist()):
                if pid not in results:
                    results[pid] = {}

                results[pid][timestamp] = (predicted_prob, predicted_label, true_label)

            for pid in results:
                results[pid] = dict(sorted(results[pid].items()))
            
    with open("../results/"+rid+".json", "w") as f:
        json.dump(results, f)
    
    # organize according to patients, use format for utility score computing
    for pid, result in tqdm(results.items()):
        df = pd.DataFrame.from_dict(result, orient='index', columns=['AllProbability', 'PredictedLabel', 'TrueLabel'])
        df['PredictedProbability'] = df.apply(lambda row: row['AllProbability'][row['PredictedLabel']], axis=1)
        result_df = df[['PredictedProbability','PredictedLabel']]
        fixed_values =  pd.DataFrame({'PredictedProbability': [0.50]*(SEQ_OFFSET-1), 'PredictedLabel': [0]*(SEQ_OFFSET-1)})
        result_df = pd.concat([fixed_values, result_df]).reset_index(drop=True)

        p = get_patient_by_id_original(int(pid))
        label_df = p['SepsisLabel']

        assert len(label_df) == len(result_df), 'pid {}'.format(pid)

        result_df.to_csv('../results/'+rid+'_pred/p'+str(pid).zfill(6)+'.psv', mode='w+', index=False, header=True, sep='|')
        label_df.to_csv('../results/'+rid+'_label/p'+str(pid).zfill(6)+'.psv', mode='w+', index=False, header=True, sep='|')

