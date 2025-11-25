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
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.log import log
from cvxpy.expressions.variable import Variable


def geo_mean_canon(expr, args):
    """
    Canonicalization for the geometric mean function.
    We reformulate geo_mean(x) as exp(1/n * sum(log(x)))
    to form a diagonal hessian instead of a dense one.
    """
    t = Variable(expr.shape, nonneg=True)

    if args[0].value is not None:
        t.value = expr.numeric(args[0].value)
    else:
        t.value = np.ones(expr.shape)

    if expr.p is None:
        return t, [log(t) == 1/expr.size * sum(log(args[0]))]
    else:
        return t, [log(t) == multiply(expr.p/sum(expr.p), log(args[0]))]
