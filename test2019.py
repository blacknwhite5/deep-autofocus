from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import os

from previous.regression import LinearRegressionModel
from loader import coxem_dataset
from previous.resnet import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = eval('{}(num_classes=1)'.format('resnet50'))
net.load_state_dict(torch.load('./previous/pretrained/{}.pth'.format('mnm2019_with_new')))
net.to(device)

transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

coxem = coxem_dataset('./data/2020_testdata/',
                  transform=transform)

loader = DataLoader(coxem, batch_size=1, shuffle=False)
criterion = nn.MSELoss()


error_per_mag = {i : [0, 0] for i in [500, 1000, 2000, 5000, 10000]}    # {mag : [cnt, total_error]}
total_error = 0
for i, (image, (target, mag), filename) in enumerate(loader):
    image = image.to(device)
    target = target.float().to(device)
    mag = mag.float().to(device)

    with torch.no_grad():
        predict = net(image)
        mse = criterion(predict, target)
        total_error += mse
    error_per_mag[int(mag.item())][0] += 1
    error_per_mag[int(mag.item())][1] += mse

    print('{0:3} : {1: 7.4f}    {2:2.1f}    {3}'.format(i, predict.item(), target.item(), filename))
print('RMS: {}'.format(torch.sqrt(total_error/len(loader))))
print('------'*4)
for i in error_per_mag:
    cnt = error_per_mag[i][0]
    err = error_per_mag[i][1]
    print('{} : {}'.format(i, torch.sqrt(err/cnt)))


#total_error = 0
#for i, (img, (target, mag), filename) in enumerate(dataloader):
#    img = img.to(device)
#    target = target.float().to(device) 
#
#    with torch.no_grad():
#        pred = net(img)
#        loss = mse(pred, target)
#        total_error += loss
#
#    print('pred: ', pred.item(), ' score: ', target.item(), ' loss :', loss.item(), ' filename: ', filename)
#print('RMSE : ', torch.sqrt(total_error/len(dataloader)).item())
