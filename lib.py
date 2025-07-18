from json import load
from pulp import const
from linear import *
import numpy as np
from highspy import Highs
import highspy
import sys

save_dir = "test/numpy"
__all__ = ["load_linear_program"]


def parse_mps_pulp(file_path):
    import pulp

    (variables, model) = pulp.LpProblem.fromMPS(file_path, sense=pulp.LpMaximize)
    var_names = list(variables.keys())
    vars = {}
    for i, var in enumerate(var_names):
        vars[var] = i
    num_vars = len(variables)
    constraints = model.constraints.values()
    num_cons = len(constraints)

    A = np.zeros((num_cons, num_vars))
    B = np.zeros(num_cons)

    types = []

    for i, (constr_name, constr) in enumerate(model.constraints.items()):
        for variable, val in constr.items():
            A[i, vars[str(variable)]] = val
        B[i] = constr.constant * -1
        if constr.sense == pulp.LpConstraintEQ:
            types.append("E")
        elif constr.sense == pulp.LpConstraintLE:
            types.append("L")
        elif constr.sense == pulp.LpConstraintGE:
            types.append("G")
        else:
            types.append("U")  # Unknown
    C = np.zeros(num_vars)
    # # Get objective coefficients
    for var, val in model.objective.items():
        C[vars[str(var)]] = val

    A, B, types = reformat_constraints(A, B, types)
    return A, B, C, types


def reformat_constraints(A, B, types):
    new_A_rows = []
    new_B_vals = []
    new_types = []

    for i, constr_type in enumerate(types):
        if constr_type == "E":
            if B[i] < 0:
                new_A_rows.append(-A[i])
                new_B_vals.append(-B[i])
                new_types.append("G")
                new_A_rows.append(-A[i])
                new_B_vals.append(-B[i])
                new_types.append("L")
            else:
                new_A_rows.append(A[i])
                new_B_vals.append(B[i])
                new_types.append("L")
                new_A_rows.append(A[i])
                new_B_vals.append(B[i])
                new_types.append("G")
        elif constr_type == "G":
            if B[i] < 0:
                new_A_rows.append(-A[i])
                new_B_vals.append(-B[i])
                new_types.append("L")
            else:
                new_A_rows.append(A[i])
                new_B_vals.append(B[i])
                new_types.append("G")
        elif constr_type == "L":
            if B[i] < 0:
                new_A_rows.append(-A[i])
                new_B_vals.append(-B[i])
                new_types.append("G")
            else:
                new_A_rows.append(A[i])
                new_B_vals.append(B[i])
                new_types.append("L")
        else:
            print(
                f"Warning: Unknown constraint type '{constr_type}' for row {i}. Skipping."
            )

    new_A = np.array(new_A_rows, dtype=np.float64)
    new_B = np.array(new_B_vals, dtype=np.float64)

    return new_A, new_B, new_types


def load_numpy_file(file_path):
    """Load a NumPy file containing A, B, C, and types."""
    A = None
    B = None
    C = None
    types = None
    with np.load(file_path) as data:
        A = data["A"]
        B = data["B"]
        B = B.reshape(B.shape[0], 1)
        C = data["C"]
        C = C.reshape(C.shape[0], 1)
        types = data["types"]

    return A, B, C, list(types)


def parse_mps_to_standard_form(mps_path):
    from pysmps import smps_loader as smps
    import os

    A, B, C, types = parse_mps_pulp(mps_path)
    np.savez(
        (os.path.join(save_dir, os.path.basename(mps_path) + ".npz")),
        A=A,
        B=B,
        C=C,
        types=np.array(types, dtype="<U10"),
    )
    B = B.reshape(B.shape[0], 1)
    C = C.reshape(C.shape[0], 1)
    return A, B, C, types


def load_linear_program(file_path):
    if "npz" in file_path:
        return load_numpy_file(file_path)
    elif "mps" in file_path:
        return parse_mps_to_standard_form(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Only .mps and .nps files are supported."
        )
