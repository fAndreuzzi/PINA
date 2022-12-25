import argparse


def _init_parser():
    return argparse.ArgumentParser(description="Run PINA")


def _extra_enabled(args):
    return hasattr("extra", args) and args.extra


def setup_generic_run_parser(parser=None):
    if not parser:
        parser = _init_parser()

    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("id_run", help="Run ID", type=int)

    return parser


def setup_extra_features_parser(parser=None):
    if not parser:
        parser = _init_parser()

    parser.add_argument("--extra", help="Extra features", action="store_true")

    return parser


def setup_deeponet_parser(parser=None):
    if not parser:
        parser = _init_parser()

    parser.add_argument("--nobias", action="store_true")

    parser.add_argument(
        "--combos",
        help="DeepONet internal network combinations",
        type=str,
        required=True,
    )

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


def prepare_deeponet_model(args, problem, extra_feature_combo_func=None):
    combos = tuple(map(lambda combo: combo.split("-"), args.combos.split(",")))
    check_combos(combos, problem.input_variables)

    extra_feature = extra_feature_combo_func if _extra_enabled(args) else None
    networks = spawn_combo_networks(
        combos=combos,
        layers=list(map(int, args.layers.split(","))) if args.layers else [],
        output_dimension=args.hidden * len(problem.output_variables),
        func=Softplus,
        extra_feature=extra_feature,
        bias=not args.nobias,
    )

    return ComboDeepONet(
        networks,
        problem.output_variables,
        aggregator=args.aggregator,
        reduction=args.reduction,
    )
