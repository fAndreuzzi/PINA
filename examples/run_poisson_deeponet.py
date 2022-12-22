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

import argparse
import logging

logging.basicConfig(
    filename="poisson_deeponet.log", filemode="w", level=logging.INFO
)


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self, label):
        super(myFeature, self).__init__()

        if not isinstance(label, (tuple, list)):
            label = [label]
        self._label = label

    def forward(self, x):
        t = torch.sin(x.extract(self._label) * torch.pi)
        return LabelTensor(t, [f"sin({self._label})"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)

    parser.add_argument("--nobias", action="store_true")

    parser.add_argument(
        "--combos",
        help="DeepONet internal network combinations",
        type=str,
        required=True,
    )
    parser.add_argument("--extra", help="extra features", action="store_true")

    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str, default="*"
    )
    parser.add_argument(
        "--reduction", help="Reduction for DeepONet", type=str, default="+"
    )

    parser.add_argument(
        "--hidden",
        help="Number of variables in the hidden DeepONet layer",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--layers",
        help="Structure of the DeepONet partial layers",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    poisson_problem = Poisson()

    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    check_combos(combos, poisson_problem.input_variables)

    networks = spawn_combo_networks(
        combos=combos,
        layers=list(map(int, args.layers.split(","))) if args.layers else [],
        output_dimension=args.hidden,
        func=Softplus,
        extra_feature=(lambda combo: [myFeature(combo)]) if args.extra else None,
        bias=not args.nobias,
    )

    model = ComboDeepONet(
        networks,
        poisson_problem.output_variables,
        aggregator=args.aggregator,
        reduction=args.reduction,
    )

    pinn = PINN(poisson_problem, model, lr=0.01, regularizer=1e-8)
    if args.s:
        pinn.span_pts(
            20, "grid", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
        )
        pinn.span_pts(20, "grid", locations=["D"])
        pinn.train(1.e-10, 100)
        pinn.save_state("pina.poisson_{}".format(args.id_run))
    else:
        pinn.load_state("pina.poisson_{}".format(args.id_run))
        plotter = Plotter()
        plotter.plot(pinn)
