#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 과거 모델, 최신 데이터
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from loader_new import coxem_dataset
from torch.utils.data import DataLoader

from resnet import resnet50 

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--net', type=str, default='resnet50', help= '[custom, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152]')
parser.add_argument('--epoch', type=int, default=1000, help= 'epoch')
parser.add_argument('--batch', type=int, default=1, help= 'batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
args = parser.parse_args()

# path
dataset = '../../data/020620+021220/'
pretrained = './pretrained/' + 'mnm2019_with_new.pth'

# hyper-params
epochs = args.epoch
batch_size = args.batch
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# pre-trained model load 
net = eval('{}(num_classes=1)'.format(args.net))
if os.path.isfile(pretrained):
    net.load_state_dict(torch.load(pretrained))
    print('parameters loaded\n')
net.to(device)

# loss function, optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# preprocess 
transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((240,320)),
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
    f = open('mnm2019_with_new.txt','w')
     
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (img, (target, mag), _) in enumerate(dataloader):
            target = target.float().to(device)
            img = img.to(device)

            outputs = net(img)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print('[{0:5d}, {1:5d}/{2:5d}] loss: {3:.6f}'.format(
                    epoch, (i+1)*batch_size, len(dataloader)*batch_size, loss))

        torch.save(net.state_dict(), pretrained)
        f.write('{:d}, {:5d} loss: {:.3f}\n'.format(epoch+1,i+1,np.sqrt(running_loss/i+1)))
    f.write('time = {:.4f}seconds\n'.format(time.time()-start))
    f.close()

    print('{} Training Finished'.format(args.net))

if __name__ == '__main__':
   main()
