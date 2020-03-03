from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import os

from loader import coxem_dataset
from resnet import resnet50
from network import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_old = resnet50(num_classes=1)
net_new = resnet18(num_classes=1)
net_old.load_state_dict(torch.load('./pretrained/{}.pth'.format('mnm2019_with_new')))
net_new.load_state_dict(torch.load('./pretrained/{}.pth'.format('mnm2020_with_new')))
net_old.to(device)
net_new.to(device)

transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop((240, 320)),
                transforms.ToTensor()])

coxem = coxem_dataset('../../data/020620+021220/',
                  transform=transform,
                  mode='test')
dataloader = DataLoader(coxem, batch_size=1, shuffle=False)
criterion = nn.MSELoss()


max_error = 0
error_per_mag_old = {i : [0, 0] for i in [500, 1000, 2000, 5000, 10000]}    # {mag : [cnt, total_error]}
error_per_mag_new = {i : [0, 0] for i in [500, 1000, 2000, 5000, 10000]}    # {mag : [cnt, total_error]}
total_error_old = 0
total_error_new = 0
for i, (image, target, mag, _) in enumerate(dataloader):
    image = image.to(device)
    target = target.float().to(device)
    mag = mag.float().to(device)

    with torch.no_grad():
        pred_old = net_old(image)
        pred_new = net_new(image, mag/max(dataloader.dataset.range_mags))

        mse_old = criterion(pred_old, target)
        mse_new = criterion(pred_new, target)
        if mse_old > max_error:
            max_error = mse_old
            max_error_idx = i
        total_error_old += mse_old
        total_error_new += mse_new
    error_per_mag_old[int(mag.item())][0] += 1
    error_per_mag_old[int(mag.item())][1] += mse_old
    error_per_mag_new[int(mag.item())][0] += 1
    error_per_mag_new[int(mag.item())][1] += mse_new

    print('{0:3} : {1: 7.4f}    {2: 07.4f}   {3:2.1f}    {4} {5}'.format(i, pred_old.item(), pred_new.item(), target.item(), int(mag.item()), _))
print('RMS_old: {}'.format(torch.sqrt(total_error_old/len(dataloader))))
print('RMS_new: {}'.format(torch.sqrt(total_error_new/len(dataloader))))
with open('../../data/020620+021220/cache.txt','r') as r:
    data = r.read().split('\n')
    idx1, idx2, idx3 = map(int, data.pop(0).split('\t'))
    data =  data[idx1+idx2:]
print('Max error sample: {0:5d} {1}'.format(max_error_idx, data[max_error_idx].split('\t')[0]))
print('2834: {}'.format(data[2834]))
print('2135: {}'.format(data[2135]))
print('2137: {}'.format(data[2137]))
print('3007: {}'.format(data[3007]))
print('2617: {}'.format(data[2617]))
print('------'*4)
for i in error_per_mag_old:
    cnt_old = error_per_mag_old[i][0]
    err_old = error_per_mag_old[i][1]
    cnt_new = error_per_mag_new[i][0]
    err_new = error_per_mag_new[i][1]
    print('{} : old: {},    new: {}'.format(i, torch.sqrt(err_old/cnt_old), torch.sqrt(err_new/cnt_new)))
