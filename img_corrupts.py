from logging import warning

import torch
import numpy as np
import torchvision.transforms as transforms 

'''
Corruption functions for image based data
Made by Ammi Beltrán, Fernanda Borja & Luciano Vidal
'''
# Label Related
def random_labels(train_data, prob):
    # data, labels = train_data
    # unique, count = torch.unique(labels)
    for i in range(len(train_data)):
        data, label = train_data[i]
        luck = torch.rand(1)
        if (luck < prob):
            original = label
            replace = label
            while(original == replace):
                replace = np.random.randint(10)
                train_data.targets[i] = replace

# Image related
'''
Use as transform.Lambda(function)
'''
def pixel_permute(img):
    # make a flat image
    img = img.view(3, -1)
    # Get permutation for all
    idx = torch.randperm(img.shape[1], generator=torch.manual_seed(69))
    print(idx[6])
    for i, ch in enumerate(img):
        ch = ch[idx].view(ch.size())
        img[i] = ch
    img = torch.reshape(img, (3, 32, 32))
    print(img.shape)
    return (img)

def random_pixel_permute(img):
    # make a flat image
    img = img.view(3, -1)
    # Get permutation for all
    idx = torch.randperm(img.shape[1])
    print(idx[6])
    for i, ch in enumerate(img):
        ch = ch[idx].view(ch.size())
        img[i] = ch
    img = torch.reshape(img, (3, 32, 32))
    print(img.shape)
    return (img)

def gaussian_pixels(img):
    # make a flat image
    img = img.view(3, -1)
    stats = torch.empty([3, 2]) # mean, var
    # print(stats, stats.shape)
    for i, ch in enumerate(img):
        stats[i][0] = torch.mean(ch, axis = 0)
        stats[i][1] = torch.var(ch, axis = 0)
        x = np.random.normal(loc=stats[i][0].item(), scale=stats[i][1].item(), size=img.shape[1])
        img[i] = torch.from_numpy(x)
    img = torch.reshape(img, (3, 32, 32))
    return img

'''
Use after dataset creation
'''
# Lineal combination
# [between two images for now]
def lineal_imgs(train_data, alpha):
    # get random idxs to permutate 
    idxs = torch.randperm(len(train_data))
    memory = train_data.data.copy()
    for i in range(len(idxs)):
        data, label = train_data[i]
        train_data.data[i] = memory[i]*alpha + train_data.data[idxs[i]]*(1 - alpha)

