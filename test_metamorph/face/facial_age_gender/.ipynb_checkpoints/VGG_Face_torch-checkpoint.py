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


VGG_Face_torch_age = nn.Sequential( # Sequential,
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

VGG_Face_torch_gender = nn.Sequential( # Sequential,
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

class VGG_ageNet(nn.Module):
    def __init__(self, pre_trained=None):
        super(VGG_ageNet, self).__init__()

        model_age = VGG_Face_torch_age
        if pre_trained is not None:
            model_age.load_state_dict(torch.load(pre_trained))

        op_list = [FC(m) if isinstance(m, nn.Sequential) else m for m in model_age.children()]

        self.pre_model = nn.Sequential(*op_list[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 8)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x 

class VGG_genderNet(nn.Module):
    def __init__(self, pre_trained=None):
        super(VGG_genderNet, self).__init__()

        model_gender = VGG_Face_torch_gender
        if pre_trained is not None:
            model_gender.load_state_dict(torch.load(pre_trained))

        op_list = [FC(m) if isinstance(m, nn.Sequential) else m for m in model_gender.children()]

        self.pre_model = nn.Sequential(*op_list[:-1])
        # self.dropout = nn.Dropout(p=0.8)
        self.classifier = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.pre_model(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x 
    
from torchvision.models import vgg11, vgg13

class VGG_ageNet_13(nn.Module):
    def __init__(self):
        super(VGG_ageNet_13, self).__init__()
        self.model = vgg13(pretrained=False)
        self.model.classifier = nn.Sequential(
								nn.Linear(25088, 4096), 
								nn.ReLU(),
								nn.Linear(4096,4096),
								nn.ReLU(), 
								nn.Linear(4096, 8))

    def forward(self, x):
        x = self.model(x)
        return x 

class VGG_genderNet_11(nn.Module):
    def __init__(self):
        super(VGG_genderNet_11, self).__init__()
        self.model = vgg11(pretrained=False)
        self.model.classifier = nn.Sequential(
								nn.Linear(25088, 4096), 
								nn.ReLU(),
								nn.Linear(4096,4096),
								nn.ReLU(), 
								nn.Linear(4096, 2))

    def forward(self, x):
        x = self.model(x)
        return x 