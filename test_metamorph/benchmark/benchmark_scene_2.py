import torch
import torch.nn as nn
import torch.nn.functional as F

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

# interpolation on (h,w) dim
class TransformHW(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW, self).__init__()
        self.input_size = input_size
        self.output_size = (output_size[2], output_size[3])
    
    def forward(self, x):
        return F.interpolate(x, size=self.output_size, mode='bilinear')
    
# 1x1 conv to relax on C dim
class TransformC(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformC, self).__init__()
        self.trans = nn.Conv2d(input_size[1], output_size[1], kernel_size=1)

    def forward(self, x):
        return self.trans(x)


class scene_origin(nn.Module):
    def __init__(self):
        super(scene_origin, self).__init__()
        self.task1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
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
        out1 = self.task1(x)
        out2 = self.task2(x)
        return out1, out2

# SA000 = LC000 = LCR000
class scene_SA000(nn.Module):
    def __init__(self):
        super(scene_SA000, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
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
        )
        self.task1 = nn.Sequential(
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
            TransformHW((1,512,14,14), (1,512,7,7)),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
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
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2

class scene_SA001(nn.Module):
    def __init__(self):
        super(scene_SA001, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
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
            #3
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 2),
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
        )
        self.task1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2
    
class scene_SA001_test(nn.Module):
    def __init__(self):
        super(scene_SA001_test, self).__init__()
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
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.task1 = nn.Sequential(
            TransformC((1,512,14,14), (1,256,14,14)),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            #3
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
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
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
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2

# LC001 = LCR001
class scene_LC001(nn.Module):
    def __init__(self):
        super(scene_LC001, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
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
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
        )
        self.task1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2

# SA002 = LC002
class scene_SA002(nn.Module):
    def __init__(self):
        super(scene_SA002, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            #4
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 2),
            BasicBlock(128, 128, 2),
            #6
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            #3
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 2),
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
        )
        self.task1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2
    
class scene_LCR002(nn.Module):
    def __init__(self):
        super(scene_LCR002, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #3
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            #4
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 2),
            BasicBlock(128, 128, 2),
            #6
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            BasicBlock(256, 256, 2),
            #3
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
        )
        self.task1 = nn.Sequential(
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2
    
class scene_sp002(nn.Module):
    def __init__(self):
        super(scene_sp002, self).__init__()
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
            #head
            AdaptiveConcatPool(),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
        ) 
        self.task1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 20),
        )
        self.task2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2