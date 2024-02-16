from pathlib import Path
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer


class Squeeze_before_Linear(nn.Module):
    def forward(self, x):
        return x[:, 0, :] if len(x.shape)==3 else x
    
class Tuple2Tensor(nn.Module):
    def forward(self, x):
        return x[0] if isinstance(x, Tuple) else x
    
class Layer_Tuple2Tensor(nn.Module):
    def __init__(self, vit_config):
        super(Layer_Tuple2Tensor, self).__init__()
        self.Layer = ViTLayer(vit_config)
        
    def forward(self, x):
        out = self.Layer(x)
        return out[0]
    
class TransformHW(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformHW, self).__init__()
        self.input_size = input_size
        if len(output_size) == 4:
            self.output_size = (output_size[2], output_size[3])
        else:
            self.output_size = (output_size[2],)
    
    def forward(self, x):
        if len(self.output_size) > 1:
            return F.interpolate(x, size=self.output_size, mode='bilinear')
        else:
            return F.interpolate(x, size=self.output_size, mode='linear')
        
class vit_origin(nn.Module):
    def __init__(self):
        super(vit_origin, self).__init__()
        vit_config_1 = AutoConfig.from_pretrained('google/vit-large-patch16-224')
        vit_config_2 = AutoConfig.from_pretrained('google/vit-base-patch16-224')
        
        self.task1 = nn.Sequential(
            ViTEmbeddings(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1), 
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            Layer_Tuple2Tensor(vit_config_1),
            nn.LayerNorm((1024,), eps=1e-12, elementwise_affine=True),
            Squeeze_before_Linear(),
            nn.Linear(in_features=1024, out_features=20, bias=True),
        )
        self.task2 = nn.Sequential(
            ViTEmbeddings(vit_config_2),
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True),
            Squeeze_before_Linear(),
            nn.Linear(in_features=768, out_features=5, bias=True),
        )
    
    def forward(self, x):
        out1 = self.task1(x)
        out2 = self.task2(x)
        return out1, out2
    
class vit_SA_t002(nn.Module):
    def __init__(self):
        super(vit_SA_t002, self).__init__()
        vit_config_1 = AutoConfig.from_pretrained('google/vit-large-patch16-224')
        vit_config_2 = AutoConfig.from_pretrained('google/vit-base-patch16-224')
        
        self.shared = nn.Sequential(
            ViTEmbeddings(vit_config_2),
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2), 
            Layer_Tuple2Tensor(vit_config_2),
            nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True),
        )
        
        self.task1 = nn.Sequential(
            TransformHW((1,128,768), (1,128,1024)),
            Squeeze_before_Linear(),
            nn.Linear(in_features=1024, out_features=20, bias=True),
        )
        self.task2 = nn.Sequential(
            Squeeze_before_Linear(),
            nn.Linear(in_features=768, out_features=5, bias=True),
        )
    
    def forward(self, x):
        out = self.shared(x)
        out1 = self.task1(out)
        out2 = self.task2(out)
        return out1, out2