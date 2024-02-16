import torch
import torch.nn as nn
from torchvision import models
from metamorph.graph.abs_graph import Graph
from metamorph.compiler.compiler import MetaMorph
from metamorph.config import set_log_file_loc

from typing import List


def parse_model(model: nn.Module) -> List[nn.Module]:
    res = []
    for layer in model.children():
        if type(layer) in MetaMorph.BASIC_OPS:
            res.append(layer)
        elif isinstance(layer, nn.Sequential):
            res.extend(parse_model(layer))
        else:
            res.append(layer)
    return res

if __name__ == "__main__":
    resnet1 = models.resnet18(pretrained=False)
    resnet2 = models.resnet18(pretrained=False)
    resnet3 = models.resnet18(pretrained=False)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    models = [parse_model(resnet1), parse_model(resnet2), parse_model(resnet3)]
    sample_input = torch.ones(1,3,224,224)
    
    set_log_file_loc(f"tmp.log")

    total_capacity, branch_capacities = 0, []
    graph = Graph(sample_input, models, device=device)
    total_capacity, branch_capacities = graph.capacity()
    print(f"total_capacity: {total_capacity}")
    for cap in branch_capacities:
        print(cap)
    print(graph)
    graph.random_connect()
    graph.build_mergeable_nodes()
    total_capacity, branch_capacities = graph.capacity()
    print(f"total_capacity: {total_capacity}")
    for cap in branch_capacities:
        print(cap)
    print(graph)
    graph.random_connect()
    graph.build_mergeable_nodes()
    total_capacity, branch_capacities = graph.capacity()
    print(f"total_capacity: {total_capacity}")
    for cap in branch_capacities:
        print(cap)
    print(graph)
    graph.random_connect()
    total_capacity, branch_capacities = graph.capacity()
    print(f"total_capacity: {total_capacity}")
    for cap in branch_capacities:
        print(cap)
    print(graph)
