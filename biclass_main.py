import sys
sys.path.insert(1, '../src')
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix

import json
from tqdm import tqdm
from datetime import datetime
import copy
import argparse

from utils import *
from evaluate import evaluate_sepsis_score
from config import *
from biclass_models import *
from dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sepsis binary classification training.")
    
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--pos_weight', type=float, default=54.5, help='Positive class weight (default: 54.5)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training (default: 25)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model_path', type=str, default=None, help='Path for pretrained model')
    parser.add_argument('--skip_eval', action='store_true', help='Flag to skip model evaluation after trainig')
    parser.add_argument('--use_val', action='store_true', help='Flag to enable train validation split')
    
    args = parser.parse_args()
    
    with open(train_ids_filepath, "r") as f:
        train_ids = json.load(f)

    with open(test_ids_filepath, "r") as f:
        test_ids = json.load(f)

    
    if args.model == 'log':
        model = LogisticRegressionModel(37, 1, args.model_path)
        dataset = RawDataset(train_ids[:100])
    elif args.model == 'mlp':
        model = MLPModel(37, 1, [256, 1024, 128], args.model_path)
        dataset = RawDataset(train_ids[:100])
    elif args.model == 'wc':
        model = WeibullCoxModel(args.model_path)
        dataset = WeibullCoxDataset(train_ids[:100])
        
    if args.use_val:    
        total_size = len(dataset)
        train_size = int(0.8 * total_size)    # 80% of data for training
        valid_size = total_size - train_size  # 20% for validation
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = None
    
    rid = model.train(train_loader, valid_loader, args.epochs, args.batch_size, args.pos_weight, args.learning_rate)
    rid = args.model+'_'+rid
    
    if not args.skip_eval:
        results, y_label, y_prob = evaluate_model(model, rid, test_ids)
        cutoff = plot_curves(rid, y_label, y_prob)
        y_pred = save_pred(results, cutoff, rid)
        print('Using cutoff {}'.format(cutoff))
        cm = confusion_matrix(y_label, y_pred)
        print("Confusion Matrix:")
        print(cm)
        label_dirpath = '../biclass_results/biclass_label/'
        pred_dirpath = '../biclass_results/'+rid+'/'
        auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score(label_dirpath,pred_dirpath)
        print('AUROC     {:.4f} \nAUPRC     {:.4f} \nAccuracy  {:.4f} \nF-measure {:.4f} \nUtility   {:.4f}\n'.format(auroc, auprc, accuracy, f_measure, utility))
    