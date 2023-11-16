import json
import math
import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
from datetime import datetime
import copy
import wandb
from tqdm import tqdm
import time

from dataset import SepsisDataset 
from transformer import TimeSeriesClassifier 


# SEQ_LEN    = 72
# SEQ_OFFSET = 24
# BATCH_SIZE = 8192
# LR         = 0.001
# EPOCHS     = 100
# INPUT_DIM  = 37
# HIDDEN_DIM = 256
# FF_DIM     = 2048
# N_CLASSES  = 2
# N_HEADS    = 32
# N_LAYERS   = 10

SEQ_LEN    = 72
SEQ_OFFSET = 24
BATCH_SIZE = 32
LR         = 0.001
EPOCHS     = 100
INPUT_DIM  = 37
HIDDEN_DIM = 64
FF_DIM     = 128
N_CLASSES  = 2
N_HEADS    = 8
N_LAYERS   = 6

preprocess_method = "standardized"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="sepsis-transformer",

    # track hyperparameters and run metadata
    config={
        "learning_rate": LR,
        "architecture": "Transformer",
        "dataset": "Competition2019",
        "epochs": EPOCHS,
        "seqence_length": SEQ_LEN,
        "preprocessing": preprocess_method,
        "positional_encoding": True,
        "time_sequence_alignment": "End",
        "feature_expand": HIDDEN_DIM,
        "heads": N_HEADS
    }
)

dataset = SepsisDataset(seq_len=SEQ_LEN, starting_offset=SEQ_OFFSET)

total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% of data for training
valid_size = int(0.15 * total_size)  # 15% for validation
test_size = total_size - train_size - valid_size  # Remaining 15% for testing

# Randomly split dataset into train, validation, and test datasets
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TimeSeriesClassifier(
    input_dim  = INPUT_DIM,
    seq_len    = SEQ_LEN,
    hidden_dim = HIDDEN_DIM,
    nheads     = N_HEADS,
    nlayers    = N_LAYERS,
    ff_dim     = FF_DIM,
    nclasses   = N_CLASSES
).to(device)

model = torch.nn.DataParallel(model)
weights = torch.tensor([1, 38], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights).to(device)
optimizer = Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print('====== Epoch {} ======'.format(epoch))
    for batch in tqdm(train_loader):
        sequences, masks, labels = batch
        sequences, masks, labels = sequences.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    print(f'Train Loss: {total_loss / len(train_dataloader)}')

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            sequences, masks, labels = batch
            sequences, masks, labels = sequences.to(device), masks.to(device), labels.to(device)
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    print(f'Validation Loss: {total_val_loss / len(val_dataloader)}')

    wandb.log({
        "train loss"     : total_loss     / len(train_dataloader),
        "Validation loss": total_val_loss / len(val_dataloader)
    })

    if (epoch+1) % 10 == 0:
        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        torch.save(
            copy.deepcopy(model).cpu().state_dict(),
            './models/{}_{}_{:.5f}.pth'.format(timestr, epoch, loss)
        )

wandb.finish()