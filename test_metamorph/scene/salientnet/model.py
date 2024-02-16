import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, vgg16_bn
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class AdaptiveConcatPool(nn.Module):
    def __init__(self, sz=(1,1)):
        super(AdaptiveConcatPool, self).__init__()
        self.sz = sz 
        self.amp = nn.AdaptiveMaxPool2d(sz)
        self.aap = nn.AdaptiveAvgPool2d(sz)
        
    def forward(self, x):
        return torch.cat((self.amp(x), self.aap(x)), dim=1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self,batch):
        return batch.view([batch.shape[0], -1])


multi_classifier_head = nn.Sequential(
                            AdaptiveConcatPool(),
                            Flatten(),
                            nn.BatchNorm1d(1024),
                            nn.Dropout(0.25),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.5),
                            nn.Linear(512, 5),
                            # nn.Sigmoid()
                        )

multi_classifier_head_vgg = nn.Sequential(
                            AdaptiveConcatPool(),
                            Flatten(),
                            nn.BatchNorm1d(1024),
                            nn.Dropout(0.25),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.5),
                            nn.Linear(512, 5),
                            # nn.Sigmoid()
                        )

def get_resnet18_model_with_custom_head(custom_head=multi_classifier_head):
    model = resnet18(pretrained=False)
    model = nn.Sequential(*list(model.children())[:-2])
    
    model.add_module('custom head', custom_head)
    # model = model.to(device)
    return model

def get_vgg16_model_with_custom_head(custom_head=multi_classifier_head_vgg):
    model = vgg16_bn(pretrained=False)
    model = nn.Sequential(*list(model.children())[:-2])

    model.add_module('custom head', custom_head)
    return model