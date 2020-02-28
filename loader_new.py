from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import random
import torch
import os

class coxem_dataset(Dataset):
    def __init__(self, path, mode='train', transform=None, normalize=False, ratio=0.9):
        self.ratio = ratio 
        self.path = path
        self.transform = transform
        self.normalize = normalize
        self.mode = mode
        self.__read_cache() if os.path.exists(os.path.join(self.path, 'cache.txt')) else self.__make_cache() 

        # do normalize?
        if self.normalize:
            self.transform.transforms.append(transforms.Normalize((0.5,), (0.5,)))

    def __getitem__(self, idx):
        img, wd, mag, specimen = self.data[idx]
        img = Image.open(img)
        img = self.transform(img)
        return img, (wd, mag), specimen

    def __len__(self):
        return len(self.data)


    def __make_cache(self):
        imgfiles = glob(os.path.join(self.path, '**/*.jpg'), recursive=True)

        # make labels
        labels = {'0':{}, 'grid':{}, 'tinball':{}}
        for i in range(11):
            labels['0'][i] = []
            labels['grid'][i] = []
            labels['tinball'][i] = []

        for imgfile in imgfiles:
            specimen = self.__what_kind(imgfile)
            with open(imgfile.replace('.jpg', '.txt'), 'r') as r:
                for content in r.read().split('\n'):
                    if 'TARGET' in content:
                        wd = int(content.split(',')[-1]) + 1    #[-1,9] --> [0,10]
                    if 'MAG' in content and 'IMAGETYPE' not in content:
                        mag = int(content.split(',')[-1])
                labels[specimen][wd].append((imgfile, wd, mag, specimen))

        # divide train/test [[train:9, valid:1]:9, test:1]
        train, valid, test = [], [], []
        for specimen in ['0', 'grid', 'tinball']:
            for i in range(11):
                data = labels[specimen][i]
                random.shuffle(data)
                div_test = int(len(data)*self.ratio)
                div_val = int(div_test*self.ratio)
                train += data[:div_val]
                valid += data[div_val:div_test]
                test += data[div_test:]
        
        with open(os.path.join(self.path, 'cache.txt'), 'w') as w:
            w.write('{}\t{}\t{}\n'.format(len(train), len(valid), len(test)))
            for content in train:
                w.write('{}\t{}\t{}\t{}\n'.format(*content))
            for content in valid:
                w.write('{}\t{}\t{}\t{}\n'.format(*content))
            for content in test:
                w.write('{}\t{}\t{}\t{}\n'.format(*content))
        self.__read_cache()

    def __what_kind(self, imgpath):
        split = imgpath.lower().split('/')[-3]
        if 'grid' in split:
            specimen = 'grid'
        elif 'tinball' in split:
            specimen = 'tinball'
        else:
            specimen = '0'
        return specimen

    def __read_cache(self):
        with open(os.path.join(self.path, 'cache.txt')) as f:
            self.data = f.read().split('\n')[:-1]
            idx1, idx2, _ = map(int, self.data.pop(0).split('\t'))

        for i in range(len(self.data)):
            self.data[i] = self.data[i].split('\t')
            # make relative path
            self.data[i][0] = os.path.join(self.path, self.data[i][0])
            # wd, mag (str --> int, float)
            self.data[i][1] = torch.tensor(int(self.data[i][1])).unsqueeze(0)
            self.data[i][2] = torch.tensor(float(self.data[i][2])).unsqueeze(0)
            # normalize magnitude
            self.data[i][2] = torch.log2(self.data[i][2]/2000) / 2.5 if self.normalize else self.data[i][2] / 10000

        # splits train/test 
        if self.mode == 'train':
            self.data = self.data[:idx1]
        elif self.mode == 'valid':
            self.data = self.data[idx1:idx1+idx2]
        elif self.mode == 'test':
            self.data = self.data[idx1+idx2:]
