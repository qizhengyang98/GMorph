import torch
import torch.nn as nn
from torch.autograd import Variable
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

class FC(nn.Module):
    def __init__(self, Linear):
        super(FC, self).__init__()
        self.Linear = Linear
    
    def forward(self, input):
        return self.Linear(input)


VGG_Face_torch = nn.Sequential( # Sequential,
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
	Lambda(lambda_fn_1), # View,
	nn.Sequential(Lambda(lambda_fn_2),nn.Linear(25088,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda_fn_2),nn.Linear(4096,4096)), # Linear,
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Sequential(Lambda(lambda_fn_2),nn.Linear(4096,2622)), # Linear,
)

class VGG_emotionNet(nn.Module):
    def __init__(self, pre_trained=None):
        super(VGG_emotionNet, self).__init__()

        model_emotion = VGG_Face_torch
        if pre_trained is not None:
            model_emotion.load_state_dict(torch.load(pre_trained))

        op_list = [FC(m) if isinstance(m, nn.Sequential) else m for m in model_emotion.children()]

        self.pre_model = nn.Sequential(*op_list[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)

        return x 


VGG_Face_torch_8 = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),

	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),

	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),

	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(512),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(512),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),

	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(512),
	nn.ReLU(),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(512),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),

	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
	nn.Dropout(0.5),
	nn.ReLU(),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
	nn.Dropout(0.5),
	nn.ReLU(),
)

class VGG_emotionNet_8(nn.Module):
    def __init__(self):
        super(VGG_emotionNet_8, self).__init__()

        self.pre_model = VGG_Face_torch_8
        self.classifier = nn.Linear(4096, 7)

    def forward(self, x):
        x = self.pre_model(x)
        x = self.classifier(x)
        return x 
    
from torchvision.models import vgg13

class VGG_emotionNet_13(nn.Module):
    def __init__(self):
        super(VGG_emotionNet_13, self).__init__()
        self.model = vgg13(pretrained=False)
        self.model.classifier = nn.Sequential(
								nn.Linear(25088, 4096), 
								nn.ReLU(),
								nn.Linear(4096,4096),
								nn.ReLU(), 
								nn.Linear(4096, 7))

    def forward(self, x):
        x = self.model(x)
        return x 