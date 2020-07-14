#####################################################################
# squeezenet.py
#
# Dev. Dongwon Paek
# Description: Main source code of SqueezeNet
#####################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

from model import squeezenet
from torchsummary import summary

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


print("SqueezeNet Summary")
model = squeezenet()
summary(model.cuda(), (3, 224, 224))

def train(model, train_loader, optimizer, epoch):
    model.train()