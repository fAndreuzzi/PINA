import argparse
import torch
from torch.nn import Softplus
from pina import Plotter, LabelTensor, PINN
from pina.model import FeedForward, ComboDeepONet, spawn_combo_networks
from problems.parametric_poisson import ParametricPoisson


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument(
        "--combos", help="DeepONet internal network combinations", type=str
    )
    parser.add_argument(
        "--aggregator", help="Aggregator for DeepONet", type=str
    )
    args = parser.parse_args()

    poisson_problem = ParametricPoisson()

    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    for combo in combos:
        for variable in combo:
            if variable not in poisson_problem.input_variables:
                raise ValueError(
                    "Combinations should be (overlapping) subsets of input variables, {} is not an input variable".format(
                        c
                    )
                )
    networks = spawn_combo_networks(combos, [10, 10, 10, 10], Softplus)
    model = ComboDeepONet(
        networks, poisson_problem.output_variables, aggregator=args.aggregator
    )

    pinn = PINN(poisson_problem, model, lr=0.006, regularizer=1e-6)
    if args.s:
        pinn.span_pts(
            {"variables": ["x", "y"], "mode": "random", "n": 100},
            {"variables": ["mu1", "mu2"], "mode": "grid", "n": 5},
            locations=["D"],
        )
        pinn.span_pts(
            {"variables": ["x", "y"], "mode": "grid", "n": 20},
            {"variables": ["mu1", "mu2"], "mode": "grid", "n": 5},
            locations=["gamma1", "gamma2", "gamma3", "gamma4"],
        )
        pinn.train(10000, 100)
        pinn.save_state("pina.poisson_param")
    else:
        pinn.load_state("pina.poisson_param")
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={"mu1": 0, "mu2": 1}, levels=21)
        plotter.plot(pinn, fixed_variables={"mu1": 1, "mu2": -1}, levels=21)
