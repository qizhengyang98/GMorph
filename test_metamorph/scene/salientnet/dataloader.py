import scipy.io as sio
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

kwargs = {'num_workers': 4, 'pin_memory': True}
bs = 128


transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])

transform_test  = transforms.Compose([transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

class ESOSDataset(Dataset):
    def __init__(self, loc='./dataset/ESOS/', split='train', transform=transform_train, target_transform=None):
        self.loc = loc
        mat = sio.loadmat(f'{self.loc}imgIdx.mat')['imgIdx'][0]
        X_train, Y_train = [], []
        X_test, Y_test = [], []
        for i in mat:
            if i[2] == 0:
                X_train.append(i[0][0])
                Y_train.append(i[1][0,0])
            else:
                X_test.append(i[0][0])
                Y_test.append(i[1][0,0])
        if split=='train':
            self.X = np.array(X_train)
            self.Y = np.array(Y_train)
        else:
            self.X = np.array(X_test)
            self.Y = np.array(Y_test)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.X[idx]
        image = Image.open(f'{self.loc}img/{img_name}')
        image = image.convert('RGB')
        y = self.Y[idx]
        y = torch.Tensor([float(y)])
        image_x = self.transform(image)
        if self.target_transform:
            y = self.target_transform(y)
        return [image_x, y]

def load_data(path, bs=bs, kwargs=kwargs):
    train_loader = torch.utils.data.DataLoader(ESOSDataset(path, 
                                                split='train', transform=transform_train), 
                                                batch_size=bs, shuffle=True, **kwargs)                                   
    test_loader = torch.utils.data.DataLoader(ESOSDataset(path, 
                                                split='test', transform=transform_test), 
                                                batch_size=bs, shuffle=False, **kwargs)
    return train_loader, test_loader


def target_transform_sampler(x):
    target = torch.tensor([0]).long()
    return target

def load_data_sampler(path):
    train_data = ESOSDataset(path, 
                            split='train', transform=transform_train, target_transform=target_transform_sampler)
    return train_data


if __name__ == '__main__':
    tr, te = load_data()
    print(len(tr.dataset), len(te.dataset))
    for x, y in iter(te):
        print(x.shape, y.shape)
        break