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
    filename="poisson_deeponet.log", filemode='w', level=logging.INFO
)


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self, label):
        super(myFeature, self).__init__()

        self._label = label

    def forward(self, x):
        t = torch.sin(x.extract([self._label]) * torch.pi)
        return LabelTensor(t, [f"sin({self._label})"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    #parser.add_argument("--extra", help="extra features", action="store_true")
    #parser.add_argument(
    #    "--combos", help="DeepONet internal network combinations", type=str
    #)
    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str
    )
    parser.add_argument("--reduction", help="Reduction for DeepONet", type=str)
    #parser.add_argument(
    #    "--hidden",
    #    help="Number of variables in the hidden DeepONet layer",
    #    type=int,
    #)
    #parser.add_argument(
    #    "--layers", help="Structure of the DeepONet partial layers", type=str
    #)
    args = parser.parse_args()

    #feat = [myFeature()] if args.extra else []

    poisson_problem = Poisson()

    # hidden_layers = list(map(int, args.layers.split(",")))
    # combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    # check_combos(combos, poisson_problem.input_variables)
    # networks = spawn_combo_networks(combos, hidden_layers, args.hidden,
    #                Softplus, extra_features=feat)

    x = FeedForward(
        layers=[],
        input_variables=["x"],
        output_variables=1,
        extra_features=[myFeature("x")],
        bias=False
    )
    y = FeedForward(
        layers=[],
        input_variables=["y"],
        output_variables=1,
        extra_features=[myFeature("y")],
        bias=False
    )
    networks = [x, y]

    model = ComboDeepONet(
        networks,
        poisson_problem.output_variables,
        aggregator=args.aggregator,
        reduction="+",
    )

    pinn = PINN(poisson_problem, model, lr=0.03, regularizer=1e-5)
    if args.s:
        pinn.span_pts(
            20, "grid", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
        )
        pinn.span_pts(20, "grid", locations=["D"])
        pinn.train(300, 100)
        pinn.save_state("pina.poisson_{}".format(args.id_run))
    else:
        pinn.load_state("pina.poisson_{}".format(args.id_run))
        plotter = Plotter()
        plotter.plot(pinn)

    logging.info('Net 0')
    logging.info(model._nets[0].model[0].weight)
    logging.info(model._nets[0].model[0].bias)

    logging.info('Net 1')
    logging.info(model._nets[1].model[0].weight)
    logging.info(model._nets[1].model[0].bias)
