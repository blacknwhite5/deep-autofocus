#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 최신 모델,  과거 데이터

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from loader import coxem_dataset
from torch.utils.data import DataLoader

from network import resnet18 
from regression import LinearRegressionModel

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--net', type=str, default='resnet18', help= '[custom, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152]')
parser.add_argument('--epoch', type=int, default=100, help= 'epoch')
parser.add_argument('--batch', type=int, default=1, help= 'batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
args = parser.parse_args()

# path
dataset = '../../data/dataset/'
pretrained = './pretrained/' + 'mnm2020_with_old.pth'
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
        transforms.CenterCrop((240, 320)),
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
         
    f = open('mnm2020_with_old.txt','w')
     
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (img, target, mag, _) in enumerate(dataloader):
            target = target.float().to(device)
            mag = mag.float().to(device) / 30000
            img = img.to(device)

            if len(target[0]) == 3:
                target = regression(target)

            outputs = net(img, mag)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print('[{0:5d}, {1:5d}/{2:5d}] loss: {3:.6f}'.format(
                    epoch, (i+1)*batch_size, len(dataloader)*batch_size, loss))

        f.write('{:d}, {:5d} loss: {:.3f}\n'.format(epoch+1,i+1,np.sqrt(running_loss/i+1)))
    f.write('time = {:.4f}seconds\n'.format(time.time()-start))
    f.close()

    torch.save(net.state_dict(), pretrained)
    print('{} Training Finished'.format(args.net))

if __name__ == '__main__':
   main()
