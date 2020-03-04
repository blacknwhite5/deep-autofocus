#!/usr/bin/env python
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch

from loader import coxem_dataset
from network import resnet18
from resnet import resnet50

import numpy as np
import time
import sys
import os


def setups(mode:dict) -> 'network, dataloader':
    net = eval('{}(num_classes=1)'.format(mode['network']))
    net = load_model(net, os.path.join('./pretrained', mode['pretrained']+'.pth'))
    net.name = mode['network']

    # loss function
    criterion = nn.MSELoss()

    # loader
    transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.CenterCrop(mode['crop_size']),
            transforms.ToTensor()
            ])
    dataloader = DataLoader(dataset=coxem_dataset(mode['dataset'], transform=transform, mode='test'),
                            batch_size=1, num_workers=2, shuffle=False)
    return net, criterion, dataloader


def load_model(net:'network', path:'path/pretrained/model') -> 'loads pretrained model':
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path))
        print('[*] {} parameters loaded'.format(net.__class__.__name__))
    else:
        raise Exception('[!] {} pretrained model not found'.format(path))
    return net


def test(net, criterion, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    maxmag = max(dataloader.dataset.range_mags)
    error_per_mag = {i : [0,0] for i in sorted(dataloader.dataset.range_mags)} # {mag : [cnt, total_error]}
    total_error = 0.0

    start = time.time()
    for i, (img, target, mag, name) in enumerate(dataloader):
        target = target.to(device)
        img = img.to(device)
        mag = mag.to(device)

        with torch.no_grad():
            outputs = net(img, mag/maxmag) if net.name == 'resnet18' else net(img)
            mse = criterion(outputs, target)
            total_error += mse
        error_per_mag[int(mag.item())][0] += 1
        error_per_mag[int(mag.item())][1] += mse
        print('{0:3} : {1: 7.4f}    {2:2.1f}    short_name: {3}'.format(i, outputs.item(), target.item(), name[-10:]))
    finish = time.time()
    print('RMSE: {}'.format(torch.sqrt(total_error/len(dataloader.dataset)).item()))
    print('time = {:.4f} seconds'.format(finish-start))
    print('Elapsed time per image: {}'.format((finish-start)/len(dataloader.dataset)))
    print('------'*4)
    for i in error_per_mag:
        try:
            cnt = error_per_mag[i][0]
            err = error_per_mag[i][1]
            print('mag : {}, loss : {}, counts: {}'.format(i, torch.sqrt(err/cnt), cnt))
        except:
            continue
    
    

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

    print('{} Test Started'.format(mode[sys.argv[1]]['pretrained']))
    test(*setups(mode[sys.argv[1]]))
    


if __name__ == '__main__':
   main()
