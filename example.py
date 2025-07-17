from linear import *
import numpy as np


if __name__ == "__main__":
    C = np.array([[4], [1], [4]])
    A = np.array([[2, 1, 1], [1, 2, 3], [2, 2, 1]])
    B = np.array([[2], [4], [8]])
    types = ["L", "L", "L"]
    assignment = LinearProgram(A, B, C, True, types).simplexSolver()
    result = (C.T @ assignment)[0]
