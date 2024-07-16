import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
import wandb
import random
import copy
import torch.nn.functional as F
import json
import numpy as np
from sklearn.model_selection import KFold

from utils import get_patient_by_id_standardized, get_patient_by_id_original
from config import *
from loss import UtilityLoss

def get_dataloaders(data, train_idx, val_idx, batch_size=256, num_workers=24):
    train_subset = Subset(data, train_idx)
    val_subset = Subset(data, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
    

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def make_layers(self, layers_config):
        layers = []

        for layer in layers_config:
            layer_type = layer.pop('type')
            layer_class = globals().get(layer_type)

            try:
                if layer_class:
                    layers.append(layer_class(**layer))
                else:
                    layer_class = getattr(nn, layer_type)
                    layers.append(layer_class(**layer))
                layer['type'] = layer_type
            
            except ValueError as err:
                print((f'Layer type {layer_type} not found in local or nn'))
    
        return layers

    def load_saved_model(self):
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location='cpu')
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            try:
                self.model.load_state_dict(cleaned_state_dict)
                print('Loaded saved model ' + self.model_path)
            except:
                print('Failed loading model {}' + self.model_path)
                sys.exit()
        else:
            print('Not using saved model.')
            
        return

    def save_model(self, model, rid, epoch, loss):
        if not os.path.exists('../models'):
            os.mkdir(dirpath)
        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        model_path = '../models/{}_{}_{:.5f}_{}.pth'.format(rid, epoch, loss, timestr)
        torch.save(copy.deepcopy(model).cpu().state_dict(), model_path)
        self.model_path = model_path
        return

    def forward(self, x, **kwargs):
        return self.model(x)
    
    def train_model(self, dataset, use_val=False, epochs=50, batch_size=256, pos_weight=54.5, lr=0.001, loss_criterion='BCE', logging=False, num_workers=24):
        if use_val:
            data_indices = np.arange(len(dataset))
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            folds = []
            for train_idx, val_idx in kf.split(data_indices):
                folds.append((train_idx, val_idx))
            num_folds = len(folds)
            train_len = len(folds[0][0])
            valid_len = len(folds[0][1])
            assert train_len + valid_len == len(dataset), "Error during train-validation split!"
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            valid_loader = None
            train_len = len(dataset)

        method = self.method
        
        # start a new wandb run to track this script
        if logging:
            run = wandb.init(
                project = wandb_project,
                config = {
                    "architecture"       : method,
                    "model_config"       : self.config,
                    "dataset"            : "Competition2019",
                    "preprocessing"      : "standardized",
                    "batch_size"         : batch_size,
                    "learning_rate"      : lr,
                    "epochs"             : epochs,
                    "pos_weight"         : pos_weight
                }
            )
            rid = method + '_' + run.name
        else:
            rid = method + '_trial'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = self.model
        model = model.to(device)
        model = torch.nn.DataParallel(model)

        if loss_criterion == 'BCE':
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device)
        elif loss_criterion == 'Utility':
            criterion = UtilityLoss(pos_weight).to(device)
        else:
            print('Error with Loss Criterion {}'.format(loss_criterion))
            sys.exit()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        try:
            for epoch in range(epochs):
                print('====== Epoch {} ======'.format(epoch))
                if use_val:
                    current_fold = epoch % num_folds
                    train_idx, val_idx = folds[current_fold]
                    train_loader, valid_loader = get_dataloaders(dataset, train_idx, val_idx, batch_size, num_workers)

                model.train()
                total_loss = 0

                for batch in tqdm(train_loader):
                    _, _, x_batch, y_batch, u_batch, _, _ = batch
                    y_batch = y_batch.unsqueeze(1)
                    if self.method == 'ResNet':
                        x_batch = x_batch.unsqueeze(1)
                    x_batch, y_batch, u_batch = x_batch.to(device), y_batch.to(device), u_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    if not loss_criterion == 'Utility':
                        loss = criterion(outputs, y_batch)
                    else:
                        loss = criterion(outputs, y_batch, u_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += loss.item()
                print(f'Train Loss: {total_loss / train_len}')

                if valid_loader:
                    # Validation loop
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for batch in valid_loader:
                            _, _, x_batch, y_batch, u_batch, _, _ = batch
                            if self.method == 'ResNet':
                                x_batch = x_batch.unsqueeze(1)
                            y_batch = y_batch.unsqueeze(1)
                            x_batch, y_batch, u_batch = x_batch.to(device), y_batch.to(device), u_batch.to(device)
                            outputs = model(x_batch)
                            if not loss_criterion == 'Utility':
                                loss = criterion(outputs, y_batch)
                            else:
                                loss = criterion(outputs, y_batch, u_batch)
                            total_val_loss += loss.item()
                    print(f'Validation Loss: {total_val_loss / valid_len}')

                if logging:
                    if valid_loader:
                        wandb.log({
                            "Train loss"     : total_loss     / train_len,
                            "Validation loss": total_val_loss / valid_len
                        })
                    else:
                        wandb.log({
                            "Train loss"     : total_loss     / train_len
                        })

                if (epoch+1) % 5 == 0:
                    epoch_loss = total_loss / train_len
                    self.save_model(model, rid, epoch, epoch_loss)

            self.load_saved_model()

        except KeyboardInterrupt:
            print("\nTraining interrupted by the user at Epoch {}.".format(epoch))
            print("Last saved {}".format(5*((epoch+1)//5)-1))

        except Exception as e:
            print("An error occurred:", str(e))
            print("Last saved {}".format(5*((epoch+1)//5)-1))
            pass

        if logging:
            wandb.finish()
        
        return rid
    
    
class LogisticRegressionModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'Log'
        self.config = config
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size[0]*input_size[1], output_size),
            nn.Sigmoid()
        )
        self.model_path = model_path
        self.load_saved_model()
        

            
class MLPModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'MLP'
        self.config = config
        self.layers = self.make_layers(config['model']['layers'])
        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResidualBlockGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride):
        super(ResidualBlockGroup, self).__init__()
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        self.group = nn.Sequential(*layers)

    def forward(self, x):
        return self.group(x)


class ResNetModel(BaseModel):
    def __init__(self, input_shape, out_classes, config, model_path):
        super(ResNetModel, self).__init__()
        self.method = 'ResNet'
        self.config = config
        
        layers_config = config['model']['layers']
        self.layers = self.make_layers(config['model']['layers'])

        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()


class TransformerModel(BaseModel):
    def __init__(self, input_size, output_size, config, model_path=None):
        super().__init__()
        self.method = 'Transformer'
        self.config = config
        self.layers = self.make_layers(self.config['model']['layers'])
        self.model = nn.Sequential(*self.layers)
        self.model_path = model_path
        self.load_saved_model()

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten the input if not already
        return self.predict(x)
            

class WeibullCoxModel(BaseModel):
    def __init__(self, model_path=None):
        super().__init__()
        self.method = 'WeibullCox'
        if model_path:
            with open(model_path, 'r') as fp:
                wc_model = json.load(fp)
            self.lambda_raw = nn.Parameter(torch.tensor([wc_model['lambda']], requires_grad=True))
            self.k_raw = nn.Parameter(torch.tensor([wc_model['k']], requires_grad=True))
            self.beta = nn.Parameter(torch.tensor(wc_model['beta'], requires_grad=True))
        else:
            self.lambda_raw = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            self.k_raw = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            self.beta = nn.Parameter(torch.randn(40, requires_grad=True))
            
    def log_likelihood(self, x, tau, S, pos_weight):
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        likelihood = (1 - S) * pos_weight * (torch.log(k) - torch.log(lambda_) + (k - 1) * torch.log(tau_lambda_ratio) + torch.matmul(x, beta)) - (tau_lambda_ratio ** k) * exp_beta_x
        return likelihood.sum()

    def forward(self, x, **param):
        tau = 6
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        probs = 1 - torch.exp(-(tau_lambda_ratio ** k) * exp_beta_x)

        return probs.squeeze()
    
    def train_model(self, dataset, use_val=False, epochs=50, batch_size=256, pos_weight=54.5, lr=0.001, loss_criterion='BCE', logging=False, num_workers=24):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam([self.lambda_raw, self.k_raw, self.beta], lr=lr)
        
        # start a new wandb run to track this script
        if logging:
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="sepsis-binary-classification",
    
                # track hyperparameters and run metadata
                config={
                    "architecture"       : "Weibull Cox",
                    "dataset"            : "Competition2019",
                    "preprocessing"      : "standardized",
                    "batch_size"         : batch_size,
                    "learning_rate"      : lr,
                    "epochs"             : epochs,
                    "pos_weight"         : pos_weight
                }
            )
            rid = run.name
        else:
            rid = self.method + '_trial'
        
        for epoch in range(epochs):
            total_loss = 0
            # patient_id, latest_hour, clinical_data, label, utility_weights, tau, S
            for _, _, x_batch, _, _, tau_batch, S_batch in tqdm(train_loader, desc="Epoch {}".format(epoch), ascii=False, ncols=75):
                optimizer.zero_grad()
                loss = -self.log_likelihood(x_batch, tau_batch, S_batch, pos_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
            if logging:
                wandb.log({
                    "Train loss" : epoch_loss
                })

        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        model_path = '../models/{}_{}_{:.5f}_{}.pth'.format(rid, epoch, epoch_loss, timestr)
        with open(model_path, "w") as fp:
            json.dump({'lambda':self.lambda_raw.item(),'k':self.k_raw.item(), 'beta':self.beta.tolist()}, fp)
            print('Model saved at:{}'.format(model_path))

        wandb.finish()
        return rid
        
        
            
    
            
    

