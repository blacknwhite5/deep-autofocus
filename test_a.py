from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import os
import time

from loader_new import coxem_dataset
from network import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = eval('{}(num_classes=1)'.format('resnet18'))
net.load_state_dict(torch.load('./pretrained/{}.pth'.format('mnm2020_with_new')))
net.to(device)

transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop((240, 320)),
                transforms.ToTensor()])

coxem = coxem_dataset('../../data/020620+021220/',
                  transform=transform,
                  mode='test')
dataloader = DataLoader(coxem, batch_size=1, shuffle=False)
criterion = nn.MSELoss()


error_per_mag = {i : [0, 0] for i in [500, 1000, 2000, 5000, 10000]}    # {mag : [cnt, total_error]}
total_error = 0
start = time.time()
for i, (image, (target, mag), _) in enumerate(dataloader):
    image = image.to(device)
    target = target.float().to(device)
    mag = mag.float().to(device)

    with torch.no_grad():
        predict = net(image, mag)
        mse = criterion(predict, target)
        total_error += mse
    error_per_mag[int(mag.item()*10000)][0] += 1
    error_per_mag[int(mag.item()*10000)][1] += mse

    print('{0:3} : {1: 7.4f}    {2:2.1f}'.format(i, predict.item(), target.item()))
print('RMS: {}'.format(torch.sqrt(total_error/len(dataloader))))
print('elapsed time: {}'.format(time.time()-start))
print('elapsed time per image: {}'.format((time.time()-start)/len(dataloader)))
print('------'*4)
for i in error_per_mag:
    cnt = error_per_mag[i][0]
    err = error_per_mag[i][1]
    print('{} : {}'.format(i, torch.sqrt(err/cnt)))
