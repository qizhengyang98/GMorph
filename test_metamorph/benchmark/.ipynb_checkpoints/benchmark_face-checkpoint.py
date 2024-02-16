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
        return output if output else input

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

class face_origin(nn.Module):
    def __init__(self):
        super(face_origin, self).__init__()
        self.task1 = nn.Sequential(
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
            
            # FC(Lambda(lambda_fn_1)), # View,
            # FC(nn.Sequential(Lambda(lambda_fn_2),nn.Linear(25088,4096))), # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # FC(nn.Sequential(Lambda(lambda_fn_2),nn.Linear(4096,4096))), # Linear,
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 8),
        )
        self.task3 = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
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
        out1 = self.task1(x)
        out2 = self.task2(x)
        out3 = self.task3(x)
        return out1, out2, out3
    
class face_half_conv(nn.Module):
    def __init__(self):
        super(face_half_conv, self).__init__()
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
    
class face_all_conv(nn.Module):
    def __init__(self):
        super(face_all_conv, self).__init__()
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
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
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

class face_SA(nn.Module):
    def __init__(self):
        super(face_SA, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.shared23 = nn.Sequential(
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
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
        shared23 = self.shared23(shared)
        out1 = self.task1(shared)
        out2 = self.task2(shared23)
        out3 = self.task3(shared23)
        return out1, out2, out3

class face_best_LC_norule(nn.Module):
    def __init__(self):
        super(face_best_LC_norule, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.shared12 = nn.Sequential(
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
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

class face_best_LC_rule(nn.Module):
    def __init__(self):
        super(face_best_LC_rule, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.shared12 = nn.Sequential(
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.task1 = nn.Sequential(
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
    
class face_SA_t002(nn.Module):
    def __init__(self):
        super(face_SA_t002, self).__init__()
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
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.shared13 = nn.Sequential(
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
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
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
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
        shared13 = self.shared13(shared)
        out1 = self.task1(shared13)
        out2 = self.task2(shared)
        out3 = self.task3(shared13)
        return out1, out2, out3
    
class face_best_LC_norule_t0(nn.Module):
    def __init__(self):
        super(face_best_LC_norule_t0, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.shared13 = nn.Sequential(
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
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
        shared13 = self.shared13(shared)
        out1 = self.task1(shared13)
        out2 = self.task2(shared)
        out3 = self.task3(shared13)
        return out1, out2, out3
    
class face_best_LC_rule_t0(nn.Module):
    def __init__(self):
        super(face_best_LC_rule_t0, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.shared12 = nn.Sequential(
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
    
class face_best_LC_norule_t002(nn.Module):
    def __init__(self):
        super(face_best_LC_norule_t002, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
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
        )
        self.task1 = nn.Sequential(
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
        out1 = self.task1(shared)
        out2 = self.task2(shared)
        out3 = self.task3(shared)
        return out1, out2, out3