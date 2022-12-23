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
    combos, layers, output_dimension, func, extra_feature, **ff_kwargs
):
    if not ComboDeepONet.is_function(extra_feature):
        extra_feature_func = lambda _: extra_feature
    else:
        extra_feature_func = extra_feature

    return [
        FeedForward(
            layers=layers,
            input_variables=tuple(combo),
            output_variables=output_dimension,
            func=func,
            extra_features=extra_feature_func(combo),
            **ff_kwargs,
        )
        for combo in combos
    ]


class ComboDeepONet(torch.nn.Module):
    def __init__(self, nets, output_variables, aggregator="+", reduction="+"):
        super().__init__()

        self.output_variables = output_variables
        self.output_dimension = len(output_variables)

        self._init_aggregator(aggregator)
        hidden_size = nets[0].model[-1].out_features
        self._init_reduction(reduction, hidden_size=hidden_size)

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

    @staticmethod
    def is_function(f):
        return type(f) == types.FunctionType or type(f) == types.LambdaType

    def _init_aggregator(self, aggregator):
        aggregator_funcs = ComboDeepONet._symbol_functions(dim=0)
        if aggregator in aggregator_funcs:
            aggregator_func = aggregator_funcs[aggregator]
        elif isinstance(
            aggregator, torch.nn.Module
        ) or ComboDeepONet.is_function(aggregator):
            aggregator_func = aggregator
        else:
            raise ValueError(f"Unsupported aggregation type {type(reduction)}")

        self._aggregator = aggregator_func
        logging.info(f"Selected aggregator: {str(aggregator_func)}")

        # test the aggregator
        if self._aggregator(torch.ones((2, 20, 3))).shape != (20, 3):
            raise ValueError("Invalid output shape for the given aggregator")

    def _init_reduction(self, reduction, hidden_size):
        reduction_funcs = ComboDeepONet._symbol_functions(dim=2)
        if reduction in reduction_funcs:
            reduction_func = reduction_funcs[reduction]
        elif isinstance(
            reduction, torch.nn.Module
        ) or ComboDeepONet.is_function(reduction):
            reduction_func = reduction
        elif reduction == "linear":
            reduction_func = nn.Linear(hidden_size, len(self.output_variables))
        else:
            raise ValueError(f"Unsupported reduction type {type(reduction)}")

        self._reduction = reduction_func
        logging.info(f"Selected reduction: {str(reduction)}")

        # test the reduction
        test = self._reduction(torch.ones((20, 3, hidden_size)))
        if test.ndim < 2 or tuple(test.shape)[:2] != (20, 3):
            raise ValueError("Invalid output shape for the given reduction")

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
        aggregated = self._aggregator(torch.stack(nets_outputs))
        aggregated_reshaped = aggregated.view(
            (len(x), len(self.output_variables), -1)
        )
        output_ = self._reduction(aggregated_reshaped)
        if output_.ndim == 3 and output_.shape[2] == 1:
            output_ = output_[..., 0]

        assert output_.shape == (len(x), len(self.output_variables))

        output_ = output_.as_subclass(LabelTensor)
        output_.labels = self.output_variables
        return output_
