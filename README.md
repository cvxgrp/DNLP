DNLP (Disciplined Nonlinear Programming)
=====================
An extension of CVXPY for general smooth nonlinear optimization.

**Contents**
- [Overview](#overview)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Supported Solvers](#supported-solvers)
- [Citing](#citing)

## Overview

DNLP extends [CVXPY](https://www.cvxpy.org/) with **Disciplined Nonlinear Programming** support.

With DNLP, you can model:
* convex optimization problems (via CVXPY's DCP),
* mixed-integer convex optimization problems,
* geometric programs (DGP),
* quasiconvex programs (DQCP), and
* **smooth nonlinear programs (DNLP)**.

DNLP uses the same expressive syntax as CVXPY, allowing you to write your problem in a natural way that follows the math.

### Example

```python
import cvxpy as cp
import numpy as np

# Problem data
n = 5
np.random.seed(1)
c = np.random.randn(n)

# Define variable
x = cp.Variable(n)

# Nonlinear objective: sum of logs
objective = cp.Maximize(cp.sum(cp.log(x)) + c @ x)

# Constraints
constraints = [
    cp.sum(x) == 1,
    x >= 0.01
]

# Create and solve problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.IPOPT)

print(f"Optimal value: {prob.value}")
print(f"Optimal x: {x.value}")
```

## Installation

### Step 1: Install IPOPT via Conda

The recommended solver is IPOPT, which can be installed along with its Python interface `cyipopt`:

```bash
conda install -c conda-forge cyipopt
```

This will install both the solver and the Python bindings.

### Step 2: Install DNLP

DNLP is installed by cloning this repository and installing locally:

```bash
git clone https://github.com/cvxgrp/DNLP.git
cd DNLP
pip install .
```

For development mode run the following command instead:

```bash
pip install -e .
```

## Examples

Many examples from various fields such as finance, energy, signal processing, machine learning, etc.,
can be found in the [dnlp-examples](https://github.com/cvxgrp/dnlp-examples)

## Supported Solvers

DNLP supports the following NLP solvers:

| Solver | License | Installation |
|--------|---------|--------------|
| [IPOPT](https://github.com/coin-or/Ipopt) | EPL-2.0 | `conda install -c conda-forge cyipopt` |
| [Knitro](https://www.artelys.com/solvers/knitro/) | Commercial | Requires license |

## Citing

If you use DNLP for academic work, we encourage you to cite the DNLP paper

```bibtex
@article{cederberg2025dnlp,
  title={Disciplined Nonlinear Programming},
  author={Daniel Cederberg, William Zhang, Parth Nobel and Stephen Boyd},
  journal={},
  volume={},
  number={},
  pages={},
  year={2025}
}
```
