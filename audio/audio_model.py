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
Models
'''
# Old Model Class
# Made for stereo!!!
class OldAudioModel(nn.Module):
    def __init__(self):
        super(OldAudioModel, self).__init__()
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
            nn.Flatten(),
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

# New simpler Audio Model, only MLP's and the size equals the batch size  

class AudioModel(nn.Module):
    def __init__(self, input_size = 22016, hidden_size = 512, output_size = 10):
        super(AudioModel, self).__init__() 
    # def layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_features = input_size, out_features = hidden_size),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features = hidden_size, out_features = hidden_size),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features = hidden_size, out_features = output_size),
        )

        self.layers = [self.layer1, self.layer2, self.layer3]
        # Init weights
        for layer in self.layers:
            layer.apply(init_weights)

    def encode(self, x, depth = 3):
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
    # print(y)
    model.train()
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    zeros = torch.zeros_like(y_pred)
    # print(zeros.size())
    for i, pred in enumerate(y):
        zeros[i, pred] = 1
    # Check if return value is labels lenght or a single number
    # Assuming the first:
    # y_pred = torch.argmax(y_pred, dim = 0).to(device)
    loss = criterion(y_pred, zeros)
    del x, zeros
    torch.cuda.empty_cache()
    return loss, y, y_pred

def train_epoch(model, train_dataset, criterion, optimizer, device = "cuda"):
    acc = 0
    t_loss = 0
    count = 0
    total_prediction = 0
    for i, batch in enumerate(train_dataset):
        # print(f"Batch {i}, batch[0][0][0][0] = {batch[0][0][0][0]}")
        loss, y, y_pred = get_loss(model, batch, criterion, device)
        #
        optimizer.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        #
        t_loss +=loss
        # print(y)
        # print(y_pred)
        _, prediction = torch.max(y_pred, 1)
        acc +=(y == prediction).sum().item()
        # count +=1
        total_prediction += prediction.shape[0]
        print(f"Iter: {i + 1}/{len(train_dataset)}, Loss:{loss}")
        #
        del batch, loss
        torch.cuda.empty_cache()
    num_batches = len(train_dataset)
    # acc = acc/count
    acc = acc/total_prediction
    # t_loss = t_loss/count
    t_loss = t_loss/num_batches
    return t_loss, acc

def validate(model, val_dataset, criterion, device = "cuda"):
    acc = 0
    v_loss = 0
    count = 0
    total_prediction = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_dataset):
            loss, y, y_pred = get_loss(model, batch, criterion, device)
            v_loss +=loss
            _, prediction = torch.max(y_pred, 1)
            acc +=(y == prediction).sum().item()
            total_prediction += prediction.shape[0]
            del loss, batch
            torch.cuda.empty_cache()

        num_batches = len(val_dataset)
        # acc = acc/count
        acc = acc/total_prediction
        # t_loss = t_loss/count
        v_loss = v_loss/num_batches
    return v_loss, acc

def train(model, epochs, train_dataset, val_dataset, criterion, optimizer, state = None, name = "", dir = "", device = "cuda"):
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
    for epoch in range(state["epoch"], epochs):
        # torch.manual_seed(0)
        # np.random.seed(0)
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
            torch.save(model.state_dict(), os.path.join(dir, f'bestParams{name}.pt'))
        # Update state
        state["loss"][0].append(t_loss)
        state["loss"][1].append(v_loss)
        state["acc"][0].append(t_acc)
        state["acc"][1].append(v_acc)
        state["epoch"] = epoch + 1
        state["params"] = model.state_dict()
        state["bestloss"] = best_loss
        # Save last just in case, [includes loss!!!!]
        torch.save(state, os.path.join(dir, f"{name}_{epoch + 1}.pt"))
    return state["loss"], state["acc"]