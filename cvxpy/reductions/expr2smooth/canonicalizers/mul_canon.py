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
from cvxpy.expressions.variable import Variable


def mul_canon(expr, args):
    if args[0].is_constant() or args[1].is_constant():
        return expr, []
    elif isinstance(args[0], Variable) and isinstance(args[1], Variable):
        out = Variable(expr.shape)
        return out, [out == expr.copy([args[0], args[1]])]
    elif args[0].is_affine() and isinstance(args[1], Variable):
        w1 = Variable(args[0].shape)
        return expr.copy([w1, args[1]]), [w1 == args[0]]
    elif isinstance(args[0], Variable) and args[1].is_affine():
        w2 = Variable(args[1].shape)
        out = Variable(expr.shape)
        return out, [w2 == args[1], out == expr.copy([args[0], w2])]
    elif args[0].is_affine() and args[1].is_affine():
        left = Variable(args[0].shape)
        right = Variable(args[1].shape)
        out = Variable(expr.shape)
        return out, [
            left == args[0], right == args[1], out == expr.copy([left, right])
        ]
