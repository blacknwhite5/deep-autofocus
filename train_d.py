#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from loader import coxem_dataset
from torch.utils.data import DataLoader

from resnet import resnet50
from regression import LinearRegressionModel

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--net', type=str, default='resnet50', help= '[custom, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152]')
parser.add_argument('--epoch', type=int, default=100, help= 'epoch')
parser.add_argument('--batch', type=int, default=1, help= 'batch size')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
args = parser.parse_args()

# path
dataset = '../../data/dataset'
pretrained = './pretrained/' + 'mnm2019_with_old.pth'
LR = './pretrained/LinearRegression.pth'

# hyper-params
epochs = args.epoch
batch_size = args.batch
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pre-trained model load 
net = eval('{}(num_classes=1)'.format(args.net))
if os.path.isfile(pretrained):
    net.load_state_dict(torch.load(pretrained))
    print('parameters loaded')
net.to(device)

# regression model 
regression = LinearRegressionModel()
if os.path.isfile(LR):
    regression.load_state_dict(torch.load(LR))
    print('regression parameters loaded')
regression.to(device)

# loss function, optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# preprocess 
transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ])

coxem = coxem_dataset(dataset,
                      transform=transform,
                      mode='train')

dataloader = DataLoader(dataset=coxem,
                        batch_size=batch_size,
                        num_workers=2,
                        shuffle=True)


def main():
    start = time.time()
    print('{} Training Started'.format(args.net))
         
#    f = open(args.net+'.txt','w')
    f = open('mnm2019_with_old.txt','w')
     
    for epoch in range(epochs):

        both_loss = ()
        running_loss = 0.0
        
        for i, (img, target, mag,  _) in enumerate(dataloader):
            target = target.to(device)
            target = target.float()
            
            if len(target[0]) == 3:
                target = regression(target)
            img = img.to(device)
            

            outputs = net(img)
            loss = criterion(outputs.float(), target.float()).float()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('{:d}, {:5d} loss: {:.3f}'.format(epoch+1,i+1,np.sqrt(running_loss/i+1)))
        f.write('{:d}, {:5d} loss: {:.3f}\n'.format(epoch+1,i+1,np.sqrt(running_loss/i+1)))
        both_loss += (np.sqrt(running_loss/(i+1)),)
    f.write('time = {:.4f}seconds\n'.format(time.time()-start))
    f.close()

    print('{} Training Finished'.format(args.net))
    torch.save(net.state_dict(), pretrained)

if __name__ == '__main__':
   main()
