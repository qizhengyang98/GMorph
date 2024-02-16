from typing import Tuple

import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from metamorph.misc.test_UTK import read_data
from metamorph.misc.multiNN import TridentNN
from metamorph.graph.abs_graph import Graph


def load_model(age_num, gen_num, eth_num):
    tridentNN = TridentNN(age_num, gen_num, eth_num)
    tridentNN.load_state_dict(torch.load('./model/traidentNN_epoch30.pt'))
    ageNet, genNet, ethNet = tridentNN.ageNN, tridentNN.genNN, tridentNN.ethNN
    return ageNet, genNet, ethNet

def demo_random_connect(graph: Graph):
    graph.random_connect()
    graph.build_mergeable_nodes()
    graph.random_connect()
    graph.build_mergeable_nodes()
    graph.random_connect()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader, n_class = read_data()
for i,j in train_loader:
    SAMPLE_INPUT = torch.ones(i[:1].shape).to(DEVICE)
    break
n_age, n_gen, n_eth = n_class['age_num'], n_class['gen_num'], n_class['eth_num']
models = load_model(n_age, n_gen, n_eth)

MODELS = []
valid_op_list = [nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Linear, nn.Dropout, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten]
for model in models:
    model = model.to(DEVICE)
    m_op_list = []
    for i, layer in model.named_modules():
        if type(layer) in valid_op_list:
            m_op_list.append(layer)
    MODELS.append(m_op_list)

class TestGraph(unittest.TestCase):
    @staticmethod
    def get_n_outputs(graph1: Graph, graph2: Graph) -> Tuple[int, int]:
        n_out_g1 = len(str(graph1).rsplit('\n', maxsplit=1)[-1].split('\t'))
        n_out_g2 = len(str(graph2).rsplit('\n', maxsplit=1)[-1].split('\t'))
        return n_out_g1, n_out_g2

    def test_hash(self):
        """===Testing Hash of Graphs==="""
        # print(self.shortDescription())
        for _ in range(5):
            graph1 = Graph(SAMPLE_INPUT, MODELS, device=DEVICE)
            graph2 = Graph(SAMPLE_INPUT, MODELS, device=DEVICE)
            self.assertEqual(hash(graph1), hash(graph2))

            graph2.random_connect()
            self.assertNotEqual(hash(graph1), hash(graph2))

    def test_equal(self):
        """===Testing Graph Equivalence==="""
        # print(self.shortDescription())
        for _ in range(5):
            graph1 = Graph(SAMPLE_INPUT, MODELS, device=DEVICE)
            graph2 = Graph(SAMPLE_INPUT, MODELS, device=DEVICE)
            self.assertEqual(graph1, graph2)

            graph2.random_connect()
            self.assertNotEqual(graph1, graph2)

    def test_copy(self):
        """===Testing Deepcopy of a Graph==="""
        # print(self.shortDescription())
        for _ in range(5):
            graph1 = Graph(SAMPLE_INPUT, MODELS, device=DEVICE)
            graph2 = deepcopy(graph1)
            graph2.random_connect()
            graph2.build_mergeable_nodes()
            graph2.random_connect()
            graph2.build_mergeable_nodes()
            graph2.random_connect()

            # Check ids of graph1 and graph2
            self.assertNotEqual(
                id(graph1), id(graph2),
                f'The copied graph (id={id(graph2)}) have the same id with the original graph (id={id(graph1)})!'
            )
            # Check if the number of outputs are equal
            n_out_g1, n_out_g2 = self.get_n_outputs(graph1, graph2)
            self.assertEqual(
                n_out_g1, n_out_g2,
                f'The copied graph (n={n_out_g2}) does not have the same number of outputs as the original graph(n={n_out_g1})!'
            )
            # Check if the graphs are equal
            self.assertNotEqual(graph1, graph2, 'The copied graph is not equivalent to the original!')
            # Check if the graphs have the same hash
            self.assertNotEqual(
                hash(graph1), hash(graph2),
                'The copied graph have a different hash compares to the original!'
            )


if __name__ == "__main__":
    unittest.main()
