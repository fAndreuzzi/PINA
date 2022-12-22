import argparse
import torch
from torch.nn import Softplus
from pina import Plotter, LabelTensor, PINN
from pina.model import FeedForward
from pina.model.combo_deeponet import (
    ComboDeepONet,
    spawn_combo_networks,
    check_combos,
)
from problems.poisson import Poisson


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi) *
             torch.sin(x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("--extra", help="extra features", action="store_true")
    parser.add_argument(
        "--combos", help="DeepONet internal network combinations", type=str
    )
    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str
    )
    parser.add_argument(
        "--hidden", help="Number of variables in the hidden DeepONet layer", type=int
    )
    parser.add_argument(
        "--layers", help="Structure of the DeepONet partial layers", type=str
    )
    args = parser.parse_args()

    feat = [myFeature()] if args.extra else []

    poisson_problem = Poisson()

    hidden_layers = list(map(int, args.layers.split(',')))
    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    check_combos(combos, poisson_problem.input_variables)
    networks = spawn_combo_networks(combos, hidden_layers, args.hidden,
                    Softplus, extra_features=feat)
    print(networks)

    model = ComboDeepONet(
        networks, poisson_problem.output_variables, aggregator=args.aggregator
    )

    pinn = PINN(poisson_problem, model, lr=0.03, regularizer=1e-8)
    if args.s:
        pinn.span_pts(20, 'grid', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.span_pts(20, 'grid', locations=['D'])
        pinn.train(5000, 100)
        pinn.save_state("pina.poisson_{}".format(args.id_run))
    else:
        pinn.load_state("pina.poisson_{}".format(args.id_run))
        plotter = Plotter()
        plotter.plot(pinn)
