#!/usr/bin/env python
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch

from loader import coxem_dataset
from network import resnet18
from resnet import resnet50

import numpy as np
import time
import sys
import os


def setups(mode:dict) -> 'network, criterion, optimizer, dataloader':
    net = eval('{}(num_classes=1)'.format(mode['network']))
    net = load_model(net, os.path.join('./pretrained', mode['pretrained']+'.pth'))

    # loss function, optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)

    # loader
    transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(mode['crop_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])

    dataloader = {'train': DataLoader(dataset=coxem_dataset(mode['dataset'], transform=transform, mode='train'),
                                      batch_size=1, num_workers=2, shuffle=True),
                  'test': DataLoader(dataset=coxem_dataset(mode['dataset'], transform=transform, mode='test'),
                                     batch_size=1, num_workers=2, shuffle=True)}
    return net, criterion, optimizer, dataloader


def load_model(net:'network', path:'path/pretrained/model') -> 'loads pretrained model':
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path))
        print('[*] {} parameters loaded'.format(net.__class__.__name__))
    return net


class Trainer:
    def __init__(self, net, criterion, optimizer, dataloader, netname):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.netname = netname
        self.maxmag = max(dataloader['train'].dataset.range_mags)
        self.epoch = 0

    def learn(self):
        running_loss = 0.0
        for i, (img, target, mag, name) in enumerate(self.dataloader['train']):
            target = target.to(self.device)
            img = img.to(self.device)
            mag = mag.to(self.device) / self.maxmag

            outputs = self.net(img, mag) if self.netname == 'resnet18' else self.net(img)
            loss = self.criterion(outputs, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if (i+1) % 100 == 0:
                print('[{0:5d}, {1:5d}/{2:5d}] loss: {3:.6f}'.format(
                    self.epoch, i+1, len(self.dataloader['train'].dataset), np.sqrt(running_loss/i+1)))
        self.epoch += 1

    def test(self):
        running_loss = 0.0
        for i, (img, target, mag, name) in enumerate(self.dataloader['test']):
            target = target.to(self.device)
            img = img.to(self.device)
            mag = mag.to(self.device) / self.maxmag

            with torch.no_grad():
                outputs = self.net(img, mag) if self.netname == 'resnet18' else self.net(img)
                loss = self.criterion(outputs, target)
            running_loss += loss.item()

        print('[test] loss: {:.3f}'.format(np.sqrt(running_loss/len(self.dataloader['test']))))
    

def main():
    # TODO: You must set the path of the datasets with cache.txt (new/old)
    # ----------------- here ------------------------------
    new = '../../data/020620+021220/'
    old = '../../data/dataset/'
    # -----------------------------------------------------
    mode = {'a':{'network':'resnet18', 'dataset':new, 'crop_size': (240, 320),'pretrained':'mnm2020_with_new'}, 
            'b':{'network':'resnet18', 'dataset':old, 'crop_size': (224, 224), 'pretrained':'mnm2020_with_old'}, 
            'c':{'network':'resnet50', 'dataset':new, 'crop_size': (240, 320),'pretrained':'mnm2019_with_new'}, 
            'd':{'network':'resnet50', 'dataset':old, 'crop_size': (224, 224), 'pretrained':'mnm2019_with_old'}}

    netname = mode[sys.argv[1]]['network']
    pretrained = os.path.join('./pretrained/', mode[sys.argv[1]]['pretrained'])
    model = Trainer(*setups(mode[sys.argv[1]]), netname)

    print('{} Training Started'.format(mode['pretrained']))
    start = time.time()
    for epoch in range(100):
        model.learn()
        model.test()

    torch.save(model.net.state_dict(), pretrained)
    print('time = {:.4f}seconds\n'.format(time.time()-start))


if __name__ == '__main__':
   main()
