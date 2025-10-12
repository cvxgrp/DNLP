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

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.expressions.variable import Variable

MIN_BOUND = 1e-4

# We canonicalize div(f(x), g(x)) as z * y = f(x), y = g(x), y >= 0.
# In other words, it assumes that the denominator is nonnegative.
# TODO (DCED):
#  1.  is it necessary to add the bound y >= 0? Does it help in
#      terms of robustness?
def div_canon(expr, args):
    #dim_z = (1, ) if args[0].shape == () else args[0].shape
    #dim_y = (1, ) if args[1].shape == () else args[1].shape
    dim_z = args[0].shape
    dim_y = args[1].shape
    sgn_z = args[0].sign

    if sgn_z == 'NONNEGATIVE':
        z = Variable(dim_z, bounds=[0, None])
    elif sgn_z == 'NONPOSITIVE':
        z = Variable(dim_z, bounds=[None, 0])
    else:
        z = Variable(dim_z)

    y = Variable(dim_y, bounds=[0, None])

    if args[1].value is not None and np.min(args[1].value) > MIN_BOUND:
        y.value = args[1].value
    else:
        y.value = expr.point_in_domain().reshape(dim_y)

    if args[0].value is not None:
        z.value = (args[0].value / y.value).reshape(dim_z)
    else:
        z.value = expr.point_in_domain().reshape(dim_z)
    return z, [multiply(z, y) == args[0], y == args[1]]