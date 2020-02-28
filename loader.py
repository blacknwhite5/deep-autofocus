from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import random
import torch
import os

class coxem_dataset(Dataset):
    def __init__(self, path, transform=None, normalize=False):
        self.path = path
        self.transform = transform
        self.normalize = normalize

        # do normalize?
        if self.normalize:
            transform.transforms.append(transforms.Normalize((0.5,), (0.5,)))

        self.imgfiles = glob(os.path.join(path, '**/*.jpg'), recursive=True)

    def __getitem__(self, idx):
        filename = self.imgfiles[idx]
        img = Image.open(self.imgfiles[idx])
        txt = self.imgfiles[idx].replace('.jpg', '.txt')
        with open(txt, 'r') as r:
            for content in r.read().split('\n'):
                if 'TARGET' in content:
                    wd = int(content.split(',')[-1])
                if 'MAG' in content and 'IMAGETYPE' not in content:
                    mag = float(content.split(',')[-1])
        img = self.transform(img)
        return img, (wd, mag), filename

    def __len__(self):
        return len(self.imgfiles)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                ])

    path = './2019_testdata/test/'
    dataset = coxem_dataset(path,
                            transform=transform,
                            normalize=True)

    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=2,
                        shuffle=True)

    print(len(loader))
    for i, (img, (wd, mag)) in enumerate(loader):
        print(i, img.shape, wd, mag)
