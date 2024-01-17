from config import *
from prepare import train_test_split
from train import train
from test import test
from evaluate import evaluate_sepsis_score
import torch

import wandb

import argparse, json, os, sys, time, utils

from transformer import TimeSeriesClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sepsis Early Detection')
    
    if not (os.path.exists(train_ids_filepath) and os.path.exists(test_ids_filepath)):
        print(f"Generating new train test split.")
        train_test_split()
        
    with open("train_ids.json", "r") as f:
        train_ids = json.load(f)
        
    with open("test_ids.json", "r") as f:
        test_ids = json.load(f)
        
        
    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="sepsis-transformer",

        # track hyperparameters and run metadata
        config={
            "architecture"       : "Transformer",
            "dataset"            : "Competition2019",
            "preprocessing"      : preprocess_method,
            "seqence_length"     : SEQ_LEN,
            "seqence_offset"     : SEQ_OFFSET,
            "seq_alignment"      : "End",
            "batch_size"         : BATCH_SIZE,
            "learning_rate"      : LR,
            "epochs"             : EPOCHS,
            "input_dim"          : INPUT_DIM,
            "hidden_dim"         : HIDDEN_DIM,
            "feedforward_dim"    : FF_DIM,
            "n_heads"            : N_HEADS,
            "n_layers"           : N_LAYERS,
            "n_classes"          : N_CLASSES,
            "positional_encoding": True
        }
    )
    
    rid = run.name
    
    model = TimeSeriesClassifier(
        input_dim  = INPUT_DIM,
        seq_len    = SEQ_LEN,
        hidden_dim = HIDDEN_DIM,
        nheads     = N_HEADS,
        nlayers    = N_LAYERS,
        ff_dim     = FF_DIM,
        nclasses   = N_CLASSES
    )
    
    model_path = train(train_ids, model, rid)
    
    wandb.finish()

    rid = 'breezy_bird'

    model_path = '../models/breezy-bird-159_01_15_2024_10_28_37_74_0.38415.pth'
    
    saved_model = TimeSeriesClassifier(
        input_dim  = INPUT_DIM,
        seq_len    = SEQ_LEN,
        hidden_dim = HIDDEN_DIM,
        nheads     = N_HEADS,
        nlayers    = N_LAYERS,
        ff_dim     = FF_DIM,
        nclasses   = N_CLASSES
    )
    
    state_dict = torch.load(model_path, map_location='cpu')
    cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    saved_model.load_state_dict(cleaned_state_dict)
    
    test(test_ids, saved_model, rid)
    
    auroc, auprc, accuracy, f_measure, utility = evaluate_sepsis_score('../results/'+rid+'_label/','../results/'+rid+'_pred/')
    print('AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}'.format(auroc, auprc, accuracy, f_measure, utility))

    
    
    