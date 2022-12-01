import torch

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import nabla
from pina import Span, Condition


class ParametricPoisson(SpatialProblem, ParametricProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1], 'y': [-1, 1]})
    parameter_domain = Span({'a': [-1, 1], 'b': [-1, 1]})

    def laplace_equation(input_, output_):
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - input_.extract(['a']))**2
                - 2*(input_.extract(['y']) - input_.extract(['b']))**2)
        return nabla(output_.extract(['u']), input_) - force_term

    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    conditions = {
        'gamma1': Condition(
            Span({'x': [-1, 1], 'y': 1, 'a': [-1, 1], 'b': [-1, 1]}),
            nil_dirichlet),
        'gamma2': Condition(
            Span({'x': [-1, 1], 'y': -1, 'a': [-1, 1], 'b': [-1, 1]}),
            nil_dirichlet),
        'gamma3': Condition(
            Span({'x': 1, 'y': [-1, 1], 'a': [-1, 1], 'b': [-1, 1]}),
            nil_dirichlet),
        'gamma4': Condition(
            Span({'x': -1, 'y': [-1, 1], 'a': [-1, 1], 'b': [-1, 1]}),
            nil_dirichlet),
        'D': Condition(
            Span({'x': [-1, 1], 'y': [-1, 1], 'a': [-1, 1], 'b': [-1, 1]}),
            laplace_equation),
    }
