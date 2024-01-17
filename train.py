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

from config import *

def train(train_ids, model, rid):
    dataset = SepsisDataset(train_ids, seq_len=SEQ_LEN, starting_offset=SEQ_OFFSET)
    ratio = dataset.get_ratio()
    pos_weight = int(ratio[0]/ ratio[1])
    print('neg:pos ratio {} weight {}'.format(ratio, pos_weight))
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)    # 80% of data for training
    valid_size = total_size - train_size  # 20% for validation

    # Randomly split dataset into train, validation, and test datasets
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)

    model = torch.nn.DataParallel(model)
    weights = torch.tensor([1, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    
    model_path = default_model

    # Training loop
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            print('====== Epoch {} ======'.format(epoch))
            for batch in tqdm(train_loader):
                _, _, sequences, masks, labels = batch
                masks = masks.bool()
                sequences, masks, labels = sequences.to(device), masks.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(sequences, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
            print(f'Train Loss: {total_loss / len(train_loader)}')

            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in valid_loader:
                    _, _, sequences, masks, labels = batch
                    masks = masks.bool()
                    sequences, masks, labels = sequences.to(device), masks.to(device), labels.to(device)
                    outputs = model(sequences, masks)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
            print(f'Validation Loss: {total_val_loss / len(valid_loader)}')

            wandb.log({
                "train loss"     : total_loss     / len(train_loader),
                "Validation loss": total_val_loss / len(valid_loader)
            })

            if (epoch+1) % 5 == 0:
                now = datetime.now()
                timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
                model_path = '../models/'+rid+'_{}_{}_{:.5f}.pth'.format(timestr, epoch, loss)
                torch.save(copy.deepcopy(model).cpu().state_dict(), model_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by the user at Epoch {}.".format(epoch))
        print("Last saved {}".format(5*((epoch+1)//5)-1))
        return model_path
        
    except Exception as e:
        print("An error occurred:", str(e))
        return model_path
    
    return model_path