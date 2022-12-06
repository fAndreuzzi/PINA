import argparse

from torch.nn import Softplus

from pina import PINN, Plotter, LabelTensor
from pina.model import FeedForward
from pina.model.combo_deeponet import (
    spawn_combo_networks,
    check_combos,
    ComboDeepONet,
)
from problems.heat import Heat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    parser.add_argument(
        "--combos",
        help="DeepONet internal network combinations",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str, default="*"
    )
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []

    problem = Heat()

    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    check_combos(combos, problem.input_variables)
    branch_nets = spawn_combo_networks(combos, [10, 10, 10], 10, Softplus)
    model = ComboDeepONet(
        branch_nets,
        output_variables=problem.output_variables,
        aggregator=args.aggregator,
    )

    pinn = PINN(problem, model, lr=0.001, error_norm="mse", regularizer=0)

    if args.s:
        pinn.span_pts(
            500,
            "random",
            locations=["gamma_r", "gamma_l", "t0", "gamma_t", "gamma_b"],
        )
        pinn.span_pts(5000, "random", locations=["D"])
        pinn.train(2500, 10)
        pinn.save_state("pina.heat.{}.{}".format(args.id_run, args.features))
    else:
        pinn.load_state("pina.heat.{}.{}".format(args.id_run, args.features))
        plotter = Plotter()
        for i, t in enumerate(torch.linspace(1, 2, 16)):
            plotter.plot(
                pinn,
                fixed_variables={"t": t, "mu1": 1, "mu2": 2},
                filename=f"{i}.png",
            )
