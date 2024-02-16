import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

kwargs = {'num_workers': 4, 'pin_memory': True}
bs = 128

CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )
NUM_CLASSES = len(CLASSES)

def target_transform(x):
    target = torch.zeros(NUM_CLASSES)
    anno = x['annotation']['object']
    if isinstance(anno, list):
        for obj in anno:
            target[CLASSES.index(obj['name'])] = 1
    else:
        target[CLASSES.index(anno['name'])] = 1
    return target

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])

transform_test  = transforms.Compose([transforms.Resize([224,224]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def load_data(path, bs=bs, kwargs=kwargs):
    VOC_data_train = torchvision.datasets.VOCDetection(path + "trainval/",
                                                        year='2007', image_set='trainval', download=False,
                                                        transform=transform_train, target_transform=target_transform)
    VOC_data_test = torchvision.datasets.VOCDetection(path + "test/",
                                                        year='2007', image_set='test', download=False,
                                                        transform=transform_test, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(VOC_data_train, batch_size=bs, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(VOC_data_test, batch_size=bs, shuffle=False, **kwargs)

    return train_loader, test_loader


def target_transform_sampler(x):
    target = torch.tensor([0]).long()
    return target

def load_data_sampler(path):
    VOC_data_train = torchvision.datasets.VOCDetection(path + "trainval/",
                                                        year='2007', image_set='trainval', download=False,
                                                        transform=transform_train, target_transform=target_transform_sampler)

    return VOC_data_train


if __name__ == '__main__':
    tr, te = load_data()
    len_target_tr = 0
    len_target_te = 0
    for x, y in iter(tr):
        len_target_tr += len(y)
    print(x.shape)
    for x, y in iter(te):
        len_target_te += len(y)
    print(len_target_tr, len_target_te)
