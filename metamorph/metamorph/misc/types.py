from typing import List, Tuple, Union

import torch.nn as nn
from enum import Enum
from collections import namedtuple

GraphNode = Union[str, nn.Module]

Result = namedtuple('Result', ['graph', 'latency', 'cmp_graph'])


class Direction(int, Enum):
    LEFT = 1
    RIGHT = 2
    NOT_VALID = 3


class Relation(int, Enum):
    PARENT_CHILD = 1
    CHILD_PARENT = 2
    NO_RELATION = 3
    NOT_DEFINED = 4

class FinetuneLevel(int, Enum):
    SUBGRAPH = 1
    FULLGRAPH = 2
