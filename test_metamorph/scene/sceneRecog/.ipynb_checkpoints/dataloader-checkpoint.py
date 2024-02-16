import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

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

def target_transform(x):
    target = torch.tensor([x]).long()
    return target

def load_data(path, load_train=False, bs=bs, kwargs=kwargs):
    test_data = torchvision.datasets.Places365(path,
                                                split='val', small=True, download=False,
                                                transform=transform_test, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False, **kwargs)
    if load_train:
        train_data = torchvision.datasets.Places365(path,
                                                    split='train-standard', small=True, download=True,
                                                    transform=transform_train, target_transform=target_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True, **kwargs)
        return train_loader, test_loader

    return test_loader


if __name__ == '__main__':
    load_train = False
    if load_train:
        tr, te = load_data(load_train)
    else:
        te = load_data()
    print(len(te))
    for x,y in iter(te):
        print(x.shape, y.shape)
        break
