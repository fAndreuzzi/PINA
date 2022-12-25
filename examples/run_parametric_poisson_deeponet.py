from pina import Plotter, PINN
from problems.parametric_poisson import ParametricPoisson


if __name__ == "__main__":
    # fmt: off
    args = setup_deeponet_parser(
        setup_generic_run_parser()
    ).parse_args()
    # fmt: on

    poisson_problem = ParametricPoisson()

    model = prepare_deeponet_model(
        args,
        poisson_problem,
    )
    pinn = PINN(poisson_problem, model, lr=0.006, regularizer=1e-6)
    if args.save:
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
        pinn.save_state("pina.poisson_param_{}".format(args.id_run))
    if args.load:
        pinn.load_state("pina.poisson_param_{}".format(args.id_run))
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={"mu1": 0, "mu2": 1}, levels=21)
        plotter.plot(pinn, fixed_variables={"mu1": 1, "mu2": -1}, levels=21)
