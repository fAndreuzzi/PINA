import torch

from pina.problem import (
    TimeDependentProblem,
    SpatialProblem,
    ParametricProblem,
)
from pina.operators import grad, nabla
from pina import Condition
from pina.span import Span


class Heat(TimeDependentProblem, SpatialProblem, ParametricProblem):
    output_variables = ["u"]
    spatial_domain = Span({"x1": [0, 1], "x2": [0, 1]})
    parameter_domain = Span({"mu1": [0.01, 10], "mu2": [0.01, 10]})
    temporal_domain = Span({"t": [0, 2]})

    def equation(input_, output_):
        du = grad(output_, input_, d=["t"])
        nabla_u = nabla(output_, input_, d=["x1", "x2"])

        # x1, x2, t, mu1, mu2 = input_.extract(['x1', 'x2', 't', 'mu1', 'mu2']).T
        x1 = input_.extract(["x1"])
        x2 = input_.extract(["x2"])
        t = input_.extract(["t"])
        mu1 = input_.extract(["mu1"])
        mu2 = input_.extract(["mu2"])

        pi = 2.0 * torch.pi
        force = 100.0 * torch.sin(pi * x1) * torch.sin(pi * x2) * torch.sin(
            pi * t
        ) - mu1 / mu2 * (torch.exp(mu2 * output_.extract(["u"])) - 1)
        return du - nabla_u - force

    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_.extract(["u"]) - u_expected

    def initial_condition(input_, output_):
        u_expected = 0.0
        return output_.extract(["u"]) - u_expected

    conditions = {
        "gamma_l": Condition(
            Span(
                {
                    "x1": 0,
                    "x2": [0, 1],
                    "t": [0, 2],
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            nil_dirichlet,
        ),
        "gamma_r": Condition(
            Span(
                {
                    "x1": 1,
                    "x2": [0, 1],
                    "t": [0, 2],
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            nil_dirichlet,
        ),
        "gamma_t": Condition(
            Span(
                {
                    "x1": [0, 1],
                    "x2": 1,
                    "t": [0, 2],
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            nil_dirichlet,
        ),
        "gamma_b": Condition(
            Span(
                {
                    "x1": [0, 1],
                    "x2": 0,
                    "t": [0, 2],
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            nil_dirichlet,
        ),
        "t0": Condition(
            Span(
                {
                    "x1": [0, 1],
                    "x2": [0, 1],
                    "t": 0,
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            nil_dirichlet,
        ),
        "D": Condition(
            Span(
                {
                    "x1": [0, 1],
                    "x2": [0, 1],
                    "t": [0, 2],
                    "mu1": [0.01, 10],
                    "mu2": [0.01, 10],
                }
            ),
            equation,
        ),
    }


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """

    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):
        pi = torch.pi * 2.0
        x1 = x.extract(["x1"])
        x2 = x.extract(["x2"])
        t = x.extract(["t"])
        mu1 = x.extract(["mu1"])
        mu2 = x.extract(["mu2"])
        tmp = (
            100.0 * torch.sin(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * t)
        )
        tmp = tmp.as_subclass(LabelTensor)
        tmp.labels = ["k0"]
        return tmp


class Sin(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """

    def __init__(self, label):
        super().__init__()
        self.label = label

    def forward(self, x):
        pi = torch.pi * 2.0
        tmp = torch.sin(x.extract([self.label]) * pi)
        tmp = tmp.as_subclass(LabelTensor)
        tmp.labels = [f"sin({self.label})"]
        return tmp
