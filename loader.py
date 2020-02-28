import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os
from PIL import Image

class coxem_dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', transform=None, ratio=0.7):
        self.path = path
        self.transform = transform
        self.cache = 'cache.txt'
        self.mode = mode
        self.range_mags = set()

        if os.path.exists(os.path.join(path, self.cache)):
            self.__read_cache()
        else:
            raise Exception('{} is not found!'.format(self.cache))
       
    def __getitem__(self, idx):
        img, target, mag, = self.data[idx][:3]
        short_name = img.split('/')[-1]
        img = Image.open(img)
        img = self.transform(img)
        return img, target, mag, short_name

    def __len__(self):
        return len(self.data)

    def __read_cache(self):
        with open(os.path.join(self.path, self.cache)) as f:
            self.data = f.read().split('\n')[:-1]
            idx1, idx2, idx3 = map(int, self.data.pop(0).split('\t'))

        for i in range(len(self.data)):
            self.data[i] = self.data[i].split('\t')
            # make relative path
            self.data[i][0] = os.path.join(self.path, self.data[i][0])
            # wd, mag (str, str --> float, float)
            self.data[i][1] = torch.tensor(float(self.data[i][1])).unsqueeze(0)
            self.data[i][2] = torch.tensor(float(self.data[i][2])).unsqueeze(0)
            self.range_mags.add(int(self.data[i][2].item()))

        # splits train/test 
        if self.mode == 'train':
            self.data = self.data[:idx1+idx2]
        elif self.mode == 'test':
            self.data = self.data[idx1+idx2:]
