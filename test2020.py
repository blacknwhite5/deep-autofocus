from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import os

from loader import coxem_dataset 
from previous.network import resnet18

batch_size = 1
data_path = './data/2020_testdata/'
pretrained = './previous/pretrained/mnm2020_with_new.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = resnet18(num_classes=1)
net.load_state_dict(torch.load(pretrained))
net.to(device)

transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.CenterCrop((240, 320)),
                transforms.ToTensor(),
                ])

dataset = coxem_dataset(path=data_path,
                        transform=transform)

loader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=2,
                    shuffle=False)

criterion = nn.MSELoss()


error_per_mag = {i : [0, 0] for i in [500, 1000, 2000, 5000, 10000]}    # {mag : [cnt, total_error]}
total_error = 0
for i, (image, (target, mag), filename) in enumerate(loader):
    image = image.to(device)
    target = target.float().to(device)
    mag = mag.float().to(device)

    with torch.no_grad():
        predict = net(image, mag.unsqueeze(0)/10000)
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
