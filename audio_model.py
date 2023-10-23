'''
Model and training fuctions for audio data (in mfcc)
-
Made by: Ammi Beltrán, Fernanda Borja & Luciano Vidal
'''

# Import libraries

import os
import numpy as np
import torch
from torch import nn


'''
Auxiliary functions
'''
# Permute function
class Permute(nn.Module):
    def __init__(self, a, b, c):
        super(Permute, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    def forward(self, x):
        return x.permute(self.a , self.b, self.c)
    
# Weights initilializer only for convolutionals
# Kaiming and bias = 0
def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, a = 0.1)
        model.bias.data.zero_()

'''
Model
'''
# Model Class

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        # def layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size= (5, 5), stride = (2, 2), padding = (2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size= (3, 3), stride = (2, 2), padding = (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size= (3, 3), stride = (2, 2), padding = (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= (3, 3), stride = (2, 2), padding = (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size = 1),
            nn.Linear(in_features = 64, out_features = 10),
        )

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]
        # Init weights
        for layer in self.layers:
            layer.apply(init_weights)

    def encode(self, x, depth = 5):
        out = x
        for i in range(depth):
            out = self.layers[i](out)
        return out
    
    def forward(self, x):
        # Pass through all layers
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return out
    
'''
Training
'''
def get_loss(model, batch, criterion, device = "cuda"):
    x, y = batch
    model.train()
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    # Check if return value is labels lenght or a single number
    # Assuming the first:
    y_pred = torch.argmax(y_pred, dim = 0).to(device)
    loss = criterion(y_pred, y)
    return loss, y, y_pred

def train_epoch(model, train_dataset, criterion, optimizer, device = "cuda"):
    acc = 0
    t_loss = 0
    count = 0
    for i, batch in enumerate(train_dataset):
        loss, y, y_pred = get_loss(model, batch, criterion, device)
        #
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        #
        t_loss +=loss
        acc +=(y == y_pred)
        count +=1
        print(f"Iter: {i + 1}/{len(train_dataset)}, Loss:{loss}")
    acc = acc/count
    t_loss = t_loss/count
    return t_loss, acc

def validate(model, val_dataset, criterion, device = "cuda"):
    acc = 0
    v_loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            loss, y, y_pred = get_loss(model, batch, criterion, device)
            v_loss +=loss
            acc+=(y==y_pred)
            count +=1
        acc = acc/count
        v_loss = v_loss/count
    return v_loss, acc

def train(model, epochs, train_dataset, val_dataset, criterion, optimizer, save_each = 2, state = None, name = "", device = "cuda"):
    # If not trained before
    if state == None:
        state = {
            "epoch" : 0,
            "loss" : [[], []], # [train, val]
            "acc" : [[], []], # [train, val]
            "params" : None, 
            "bestloss" : np.inf
        }
    # else, previously trained
    else:
        state = torch.load(state)
        model.load_state_dict(state["params"])
    #
    best_loss = state["bestloss"]
    # Begin iterating
    for epoch in range(0, epochs):
        # Train
        print(f"Epoch nro {epoch +1}/{epochs}")
        t_loss, t_acc = train_epoch(model, train_dataset, criterion, optimizer, device)
        # Validate
        v_loss, v_acc = validate(model, val_dataset, criterion, device)
        #
        print(f"Epoch {epoch + 1}/{epochs}: Train loss = {t_loss}, Val loss = {v_loss}, Train acc = {t_acc}, Val acc = {v_acc}")
        # Save if better loss
        if v_loss < best_loss:
            best_loss = v_loss
            print(f"Better params found in epoch = {epoch + 1}, saved params")
            torch.save(model.state_dict(), f'bestParams{name}.pt')
        # Load best model so far to proceed
        # model.load_state_dict(torch.load(f'bestDownParams.pt'))
        # Save periodically for each
        if ((epoch + 1)%save_each == 0):
            print(f"Se ha guardado la época múltiplo de {save_each}")
            torch.save(model.state_dict(), f'eachDownParams{name}_{epoch + 1}.pt')
        # Update state
        state["loss"][0].append(t_loss)
        state["loss"][1].append(v_loss)
        state["acc"][0].append(t_acc)
        state["acc"][1].append(v_acc)
        state["epoch"] = epoch + 1
        state["params"] = model.state_dict()
        state["bestloss"] = best_loss
        # Save last just in case, [includes loss!!!!]
        torch.save(state, f"LastDown{name}_{epoch + 1}.pt")
        return state["loss"], state["acc"]