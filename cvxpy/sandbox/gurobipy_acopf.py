#!/usr/bin/env python

# Copyright 2025, Gurobi Optimization, LLC

# A Power Flow Problem is illustrated on a 4 bus (node) network using an AC (Alternating Current) represention
#
#
# Zij = Impedance between i and j                          P_i      = Active Power Injected into bus i
# Yij = Admittance between i and j  (Yij = 1/Zij)          Q_i      = Reactive Power Injeced into bus i
# Yij = Gij + j Bij    (j = sqrt(-1))                      V_i      = Voltage Magnitude at bus i
#                                                          Theta_i  = Voltage Angle at bus i
#
# Real (P) and Reactive (Q) Power balance equations must be met on all buses
#
# P_i = V_i \sum_{j=1}^{N} V_j (G_{ij} \cos(\theta_i - \theta_j) + B_{ij} \sin(\theta_i - \theta_j))
#
# Q_i = V_i \sum_{j=1}^{N} V_j (G_{ij} \sin(\theta_i - \theta_j) - B_{ij} \cos(\theta_i - \theta_j))
#
#
#   (1)---(2)    Bus 1: Swing Bus. Fixed Voltage Magnitude and Angle. Variable P and Q.
#    |     |     Bus 2: Load Bus.  Fixed P and Q. Variable Voltage Magnitude and Angle.
#    |     |     Bus 3. Generation bus. Fixed Voltage Magnitude and P. Variable Voltage Angle and Q.
#   (4)---(3)    Bus 4. Load Bus.  Fixed P and Q. Variable Voltage and Angle.
#
#  Objective: minimize overall reactive power on buses 1 and 3
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import nlfunc


# Number of Buses (Nodes)
N = 4

# Conductance/susceptance components
G = np.array(
    [
        [1.7647, -0.5882, 0.0, -1.1765],
        [-0.5882, 1.5611, -0.3846, -0.5882],
        [0.0, -0.3846, 1.5611, -1.1765],
        [-1.1765, -0.5882, -1.1765, 2.9412],
    ]
)
B = np.array(
    [
        [-7.0588, 2.3529, 0.0, 4.7059],
        [2.3529, -6.629, 1.9231, 2.3529],
        [0.0, 1.9231, -6.629, 4.7059],
        [4.7059, 2.3529, 4.7059, -11.7647],
    ]
)

# Assign bounds where fixings are needed
v_lb = np.array([1.0, 0.0, 1.0, 0.0])
v_ub = np.array([1.0, 1.5, 1.0, 1.5])
P_lb = np.array([-3.0, -0.3, 0.3, -0.2])
P_ub = np.array([3.0, -0.3, 0.3, -0.2])
Q_lb = np.array([-3.0, -0.2, -3.0, -0.15])
Q_ub = np.array([3.0, -0.2, 3.0, -0.15])
theta_lb = np.array([0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])
theta_ub = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 2])

with gp.Env() as env, gp.Model("OptimalPowerFlow", env=env) as model:
    P = model.addMVar(N, name="P", lb=P_lb, ub=P_ub)  # Real power for buses
    Q = model.addMVar(N, name="Q", lb=Q_lb, ub=Q_ub)  # Reactive for buses
    v = model.addMVar(N, name="v", lb=v_lb, ub=v_ub)  # Voltage magnitude at buses

    # Voltage angle at buses. The MVar is reshaped to a column vector to
    # simplify the outer subtraction used in power balance constraints.
    theta = model.addMVar(N, name="theta", lb=theta_lb, ub=theta_ub).reshape((N, 1))

    # Minimize Reactive Power at buses 1 and 3
    model.setObjective(Q[[0, 2]].sum(), GRB.MINIMIZE)

    # Real power balance
    constr_P = model.addGenConstrNL(
        P,
        v * ((G * nlfunc.cos(theta - theta.T) + B * nlfunc.sin(theta - theta.T)) @ v),
        name="constr_P",
    )

    # Reactive power balance
    constr_Q = model.addGenConstrNL(
        Q,
        v * ((G * nlfunc.sin(theta - theta.T) - B * nlfunc.cos(theta - theta.T)) @ v),
        name="constr_Q",
    )

    model.optimize()

    # Print output table if pandas is installed
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Bus": range(1, N + 1),
                "Voltage": v.X,
                "Angle": theta.reshape(N).X * 180 / np.pi,
                "Real Power": P.X,
                "Reactive Power": Q.X,
            }
        )

        print(df)
    except ImportError:
        pass