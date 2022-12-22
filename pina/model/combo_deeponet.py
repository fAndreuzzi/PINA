"""Module for DeepONet model"""
import torch
import torch.nn as nn

from pina import LabelTensor
from pina.model import FeedForward

from functools import reduce, partial
import logging
import types


def check_combos(combos, variables):
    for combo in combos:
        for variable in combo:
            if variable not in variables:
                raise ValueError(
                    f"Combinations should be (overlapping) subsets of input variables, {variable} is not an input variable"
                )


def spawn_combo_networks(
    combos, layers, output_dimension, func, extra_features
):
    return [
        FeedForward(
            layers=layers,
            input_variables=tuple(combo),
            output_variables=output_dimension,
            func=func,
            extra_features=extra_features,
        )
        for combo in combos
    ]


class ComboDeepONet(torch.nn.Module):
    def __init__(self, nets, output_variables, aggregator="+", reduction="+"):
        super().__init__()

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        self._init_aggregator(aggregator)
        self._init_reduction(reduction)

        if not ComboDeepONet._all_nets_same_output_layer_size(nets):
            raise ValueError("All networks should have the same output size")
        self._nets = torch.nn.ModuleList(nets)
        logging.info(f"Combo DeepONet children: {list(self.children())}")

    @staticmethod
    def _symbol_functions(**kwargs):
        return {
            "+": partial(torch.sum, **kwargs),
            "*": partial(torch.prod, **kwargs),
            "mean": partial(torch.mean, **kwargs),
            "min": lambda x: torch.min(x, **kwargs).values,
            "max": lambda x: torch.max(x, **kwargs).values,
        }

    def _init_aggregator(self, aggregator_symbol):
        aggregator_funcs = ComboDeepONet._symbol_functions(dim=0)
        if aggregator_symbol not in aggregator_funcs:
            raise ValueError(f"Invalid aggregator: {aggregator_symbol}")

        self._aggregator = aggregator_funcs[aggregator_symbol]
        logging.info(f"Selected aggregator: {aggregator_symbol}")

    @staticmethod
    def is_function(f):
        return type(f) == types.FunctionType

    # TODO support n-dimensional output
    def _init_reduction(self, reduction):
        reduction_funcs = ComboDeepONet._symbol_functions(dim=1)
        if reduction in reduction_funcs:
            reduction_func = reduction_funcs[reduction]
        elif isinstance(reduction, torch.nn.Module):
            reduction_func = reduction
        elif ComboDeepONet.is_function(reduction):
            reduction_func = reduction
        else:
            raise ValueError(f"Unsupported reduction type {type(reduction)}")

        self._reduction = reduction_func
        logging.info(f"Selected reduction: {reduction}")

    @staticmethod
    def _all_nets_same_output_layer_size(nets):
        size = nets[0].layers[-1].out_features
        return all((net.layers[-1].out_features == size for net in nets[1:]))

    @property
    def input_variables(self):
        nets_input_variables = map(lambda net: net.input_variables, self._nets)
        return reduce(sum, nets_input_variables)

    def forward(self, x):
        nets_outputs = tuple(
            net(x.extract(net.input_variables)) for net in self._nets
        )
        aggregated_nets_outputs = self._aggregator(torch.stack(nets_outputs))

        output_ = self._reduction(aggregated_nets_outputs)
        if output_.ndim == 1:
            output_ = output_[:, None]
        output_ = output_.as_subclass(LabelTensor)
        output_.labels = self.output_variables
        return output_
