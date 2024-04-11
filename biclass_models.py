import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from datetime import datetime
import wandb
import random
import copy
import torch.nn.functional as F
import json

from utils import get_patient_by_id_standardized, get_patient_by_id_original
from config import *

class BaseModel:
    def __init__(self):
        pass

    def train(self, train_loader, valid_loader=None, epochs=200, batch_size=2048, pos_weight=54.5, lr=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        model = self.model.to(device)
        model = torch.nn.DataParallel(model)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="sepsis-binary-classification",

            # track hyperparameters and run metadata
            config={
                "architecture"       : "MLP",
                "dataset"            : "Competition2019",
                "preprocessing"      : "standardized",
                "batch_size"         : batch_size,
                "learning_rate"      : lr,
                "epochs"             : epochs,
                "hidden_dim"         : self.hidden_sizes,
                "pos_weight"         : pos_weight
            }
        )
        rid = run.name
        
        try:
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                print('====== Epoch {} ======'.format(epoch))
                for batch in tqdm(train_loader):
                    x_batch, y_batch = batch
                    y_batch = y_batch.unsqueeze(1)
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += loss.item()
                print(f'Train Loss: {total_loss / len(train_loader)}')
                wandb.log({
                    "Train loss"     : total_loss     / len(train_loader)
                })
                
                if valid_loader:
                    # Validation loop
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for batch in valid_loader:
                            x_batch, y_batch = batch
                            y_batch = y_batch.unsqueeze(1)
                            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                            outputs = model(x_batch)
                            loss = criterion(outputs, y_batch)
                            total_val_loss += loss.item()
                    print(f'Validation Loss: {total_val_loss / len(valid_loader)}')

                    wandb.log({
                        "Validation loss": total_val_loss / len(valid_loader)
                    })

                if (epoch+1) % 5 == 0:
                    now = datetime.now()
                    timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
                    epoch_loss = total_loss / len(train_loader)
                    model_path = '../models/'+self.method+'{}_{}_{:.5f}_{}.pth'.format(rid, epoch, epoch_loss, timestr)
                    torch.save(copy.deepcopy(model).cpu().state_dict(), model_path)
                    
            state_dict = torch.load(model_path, map_location='cpu')
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state_dict)

        except KeyboardInterrupt:
            print("\nTraining interrupted by the user at Epoch {}.".format(epoch))
            print("Last saved {}".format(5*((epoch+1)//5)-1))

        except Exception as e:
            print("An error occurred:", str(e))
            print("Last saved {}".format(5*((epoch+1)//5)-1))
            pass
        
        wandb.finish()
        return rid

    def predict(self, x, **params):
        x = x.to('cuda')
        return self.model(x)
        

class LogisticRegressionModel(BaseModel):
    def __init__(self, input_size, output_size, model_path=None):
        super().__init__()
        self.method = 'log_'
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        self.hidden_sizes = None
        
        if model_path:
            state_dict = torch.load(model_path, map_location='cpu')
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state_dict)
            print('Continued trainig of model '+model_path)


class MLPModel(BaseModel):
    def __init__(self, input_size, output_size, hidden_sizes, model_path=None):
        super().__init__()
        self.method = 'mlp_'
        self.hidden_sizes = hidden_sizes
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        
        if model_path:
            state_dict = torch.load(model_path, map_location='cpu')
            cleaned_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state_dict)
            print('Continued trainig of model '+model_path)
        
        
class WeibullCoxModel(BaseModel):
    def __init__(self, model_path=None):
        if model_path:
            with open(model_path, 'r') as fp:
                wc_model = json.load(fp)
            self.lambda_raw = torch.tensor([wc_model['lambda']], requires_grad=True)
            self.k_raw = torch.tensor([wc_model['k']], requires_grad=True)
            self.beta = torch.tensor(wc_model['beta'], requires_grad=True)
        else:
            self.lambda_raw = torch.tensor([0.0], requires_grad=True)
            self.k_raw = torch.tensor([0.0], requires_grad=True)
            self.beta = torch.randn(37, requires_grad=True)
            
    def log_likelihood(self, x, tau, S, pos_weight):
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        likelihood = (1 - S) * pos_weight * (torch.log(k) - torch.log(lambda_) + (k - 1) * torch.log(tau_lambda_ratio) + torch.matmul(x, beta)) - (tau_lambda_ratio ** k) * exp_beta_x
        return likelihood.sum()

    def predict(self, x):
        tau = 6
        x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
        lambda_ = F.softplus(self.lambda_raw)
        k = F.softplus(self.k_raw)
        beta = self.beta
        tau_lambda_ratio = tau / lambda_
        exp_beta_x = torch.exp(torch.matmul(x, beta))
        probs = 1 - torch.exp(-(tau_lambda_ratio ** k) * exp_beta_x)

        return probs
    
    def train(self, train_loader, valid_loader=None, epochs=200, batch_size=2048, pos_weight=54.5, lr=0.01):
        optimizer = torch.optim.Adam([self.lambda_raw, self.k_raw, self.beta], lr=lr)
        
        # start a new wandb run to track this script
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
        
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, tau_batch, S_batch in tqdm(train_loader, desc="Epoch {}".format(epoch), ascii=False, ncols=75):
                optimizer.zero_grad()
                loss = -self.log_likelihood(x_batch, tau_batch, S_batch, pos_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
            wandb.log({
                "Train loss" : epoch_loss
            })

        now = datetime.now()
        timestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        model_path = '../models/WeibullCox_{}_{}_{:.5f}_{}.pth'.format(rid, epoch, epoch_loss, timestr)
        with open(model_path, "w") as fp:
            json.dump({'lambda':self.lambda_raw.item(),'k':self.k_raw.item(), 'beta':self.beta.tolist()}, fp)
            print('Model saved at:{}'.format(model_path))

        wandb.finish()
        return rid
        
        
            
    
            
    

