import argparse
import logging

import torch
from torch.nn import Softplus

from pina import Plotter, LabelTensor, PINN
from pina.model import FeedForward
from pina.model.combo_deeponet import (
    ComboDeepONet,
    spawn_combo_networks,
    check_combos,
    FeedForward,
)
from problems.poisson import Poisson
from utils import (
    setup_generic_run_parser,
    setup_extra_features_parser,
    setup_deeponet_parser,
    prepare_deeponet_model,
)

logging.basicConfig(
    filename="poisson_deeponet.log", filemode="w", level=logging.INFO
)


class SinFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self, label):
        super(SinFeature, self).__init__()

        if not isinstance(label, (tuple, list)):
            label = [label]
        self._label = label

    def forward(self, x):
        t = torch.sin(x.extract(self._label) * torch.pi)
        return LabelTensor(t, [f"sin({self._label})"])


if __name__ == "__main__":
    # fmt: off
    args = setup_deeponet_parser(
        setup_extra_features_parser(
            setup_generic_run_parser()
        )
    ).parse_args()
    # fmt: on

    poisson_problem = Poisson()

    model = prepare_deeponet_model(
        args,
        problem,
        extra_feature_combo_func=lambda combo: [SinFeature(combo)],
    )
    pinn = PINN(poisson_problem, model, lr=0.01, regularizer=1e-8)
    if args.save:
        pinn.span_pts(
            20, "grid", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
        )
        pinn.span_pts(20, "grid", locations=["D"])
        pinn.train(1.0e-10, 100)
        pinn.save_state("pina.poisson_{}".format(args.id_run))
    if args.load:
        pinn.load_state("pina.poisson_{}".format(args.id_run))
        plotter = Plotter()
        plotter.plot(pinn)
