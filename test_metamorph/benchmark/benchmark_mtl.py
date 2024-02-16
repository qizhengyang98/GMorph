import torch
import torch.nn as nn
from functools import reduce


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
    
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output[0] if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def lambda_fn_1(x):
	return x.view(x.size(0),-1)

def lambda_fn_2(x):
	if 1==len(x.size()):
		return x.view(1,-1)
	else:
		return x
    
class Lambda_fn_1(nn.Module):
    def __init__(self):
        super(Lambda_fn_1, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0),-1)
    
class Lambda_fn_2(nn.Module):
    def __init__(self):
        super(Lambda_fn_2, self).__init__()
    
    def forward(self, x):
        return x

class FC(nn.Module):
    def __init__(self, Linear):
        super(FC, self).__init__()
        self.Linear = Linear
    
    def forward(self, input):
        return self.Linear(input)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))

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

class b1_mtl(nn.Module):
    def __init__(self):
        super(b1_mtl, self).__init__()
        self.shared = nn.Sequential(
            # first batch (64)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # second batch (128)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Third Batch (256)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 4-th Batch (512)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 5-th Batch (512)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.task1 = nn.Sequential(
            # FC
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )
        self.task2 = nn.Sequential(
            # FC
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
        )
        self.task3 = nn.Sequential(
            # FC
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
        )

    def forward(self, x):
        shared = self.shared(x)
        out1 = self.task1(shared)
        out2 = self.task2(shared)
        out3 = self.task3(shared)
        return out1, out2, out3
    
class b2_mtl(nn.Module):
    def __init__(self):
        super(b2_mtl, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.task1 = nn.Sequential(
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            FC(Lambda_fn_1()), # View,
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(25088,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(4096,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7),
        )
        self.task23 = nn.Sequential(
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.task2 = nn.Sequential(
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            FC(Lambda_fn_1()), # View,
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(25088,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(4096,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 8),
        )
        self.task3 = nn.Sequential(
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            FC(Lambda_fn_1()), # View,
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(25088,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(4096,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )
    
    def forward(self, x):
        shared = self.shared(x)
        out1 = self.task1(shared)
        shared23 = self.task23(shared)
        out2 = self.task2(shared23)
        out3 = self.task3(shared23)
        return out1, out2, out3

class b3_mtl(nn.Module):
    def __init__(self):
        super(b3_mtl, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.shared12 = nn.Sequential(
            nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.AdaptiveAvgPool2d((7, 7)), 
            nn.Flatten(),
            nn.Linear(25088, 4096), 
            nn.ReLU(), 
            nn.Linear(4096,4096),
            nn.ReLU(), 
            nn.Linear(4096, 7),
        )
        self.task2 = nn.Sequential(
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            FC(Lambda_fn_1()), # View,
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(25088,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            FC(nn.Sequential(Lambda_fn_2(),nn.Linear(4096,4096))), # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 8),
        )
        self.task3 = nn.Sequential(
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.AdaptiveAvgPool2d((7, 7)), 
            nn.Flatten(),
            nn.Linear(25088, 4096), 
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(), 
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        shared = self.shared(x)
        shared12 = self.shared12(shared)
        out1 = self.task1(shared12)
        out2 = self.task2(shared12)
        out3 = self.task3(shared)
        return out1, out2, out3
    
class b4_mtl(nn.Module):
    def __init__(self):
        super(b4_mtl, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
        )
        self.task1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            #4
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 2),
            BasicBlock(128, 128, 2),
            BasicBlock(128, 128, 2),
            #6
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            #3
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 2),
            BasicBlock(512, 512, 2),
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            #2
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 2),
            #2
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 2),
            #2
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 2),
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        shared = self.shared(x)
        out1 = self.task1(shared)
        out2 = self.task2(shared)
        return out1, out2