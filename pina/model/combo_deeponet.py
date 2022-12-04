"""Module for DeepONet model"""
import torch
import torch.nn as nn

from pina import LabelTensor
from pina.model import FeedForward

from functools import reduce, partial
import logging


def spawn_combo_networks(combos, layers, func):
    return [
        FeedForward(
            layers=layers[:-1],
            input_variables=tuple(combo),
            output_variables=layers[-1],
            func=func,
        )
        for combo in combos
    ]


class ComboDeepONet(torch.nn.Module):
    def __init__(self, nets, output_variables, aggregator="+"):
        super().__init__()

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        if not ComboDeepONet._all_nets_same_output_layer_size:
            raise ValueError("All networks should have the same output size")
        self._nets = nets

        self._aggregator = aggregator
        n_hidden_output = nets[0].layers[-1].out_features
        self.reduction = nn.Linear(n_hidden_output, self.output_dimension)

        aggregator_funcs = {
            "+": torch.sum,
            "*": torch.prod,
            "mean": torch.mean,
            "min": torch.min,
            "max": torch.max,
        }
        self._aggregator = partial(aggregator_funcs[aggregator], dim=0)
        logging.info("Selected aggregator: {}".format(aggregator_funcs[aggregator]))

    @staticmethod
    def _all_nets_same_output_layer_size(nets):
        size = nets[0].layers[-1].out_features
        return all([net.layers[-1].out_features == size for net in nets[1:]])

    @property
    def input_variables(self):
        nets_input_variables = map(lambda net: net.input_variables)
        return reduce(sum, nets_input_variables, self._nets)

    def forward(self, x):
        nets_outputs = [
            net(x.extract(net.input_variables)) for net in self._nets
        ]
        aggregated_nets_outputs = self._aggregator(torch.stack(nets_outputs))

        output_ = self.reduction(aggregated_nets_outputs)
        output_ = output_.as_subclass(LabelTensor)
        output_.labels = self.output_variables
        return output_
