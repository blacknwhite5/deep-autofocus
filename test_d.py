from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import os

from regression import LinearRegressionModel
from loader import coxem_dataset
from resnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = eval('{}(num_classes=1)'.format('resnet50'))
net.load_state_dict(torch.load('./pretrained/{}.pth'.format('mnm2019_with_old')))
net.to(device)

regression = LinearRegressionModel()
regression.load_state_dict(torch.load('./pretrained/LinearRegression.pth'))
regression.to(device)

transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

coxem = coxem_dataset('../../data/dataset/',
                  transform=transform,
                  mode='test')
dataloader = DataLoader(coxem, batch_size=1, shuffle=False)
mse = nn.MSELoss()

total_error = 0
for i, (img, target, mag, _) in enumerate(dataloader):
    img = img.to(device)
    target = target.float().to(device)

    if len(target[0]) == 3:
        target = regression(target)
        target = target.float()

    with torch.no_grad():
        pred = net(img)
        loss = mse(pred, target)
        total_error += loss

    print('pred: ', pred.item(), ' score: ', target.item(), ' loss :', loss.item())
print('RMSE : ', torch.sqrt(total_error/len(dataloader)).item())
