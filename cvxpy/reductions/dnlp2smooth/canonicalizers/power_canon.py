"""
Copyright 2025 CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs


def power_canon(expr, args):
    x = args[0]
    p = expr.p_rational
    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    if p == 1:
        return x, []
    # case for inv_pos
    if p == -1:
        t = Variable(shape)
        return expr.copy([t]), [t == 1/x]
    # case for square root, formulate hypograph
    if 0 < p < 1:
        w = expr.w
        t = Variable(shape)
        return t, gm_constrs(t, [x, ones], w)
    # case for square, treated as smooth
    elif p > 1:
        t = Variable(shape)
        if isinstance(args[0], Variable):
            return expr.copy(args), []

        t = Variable(args[0].shape)

        if args[0].value is not None:
            t.value = args[0].value
        else:
            t.value = expr.point_in_domain()

        return expr.copy([t]), [t==args[0]]
    else:
        raise NotImplementedError(f'The power {p} is not yet supported.')
