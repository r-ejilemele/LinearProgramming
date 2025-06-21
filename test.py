from linear import *
import numpy as np

A = np.array([[3, 2, 1], [2, 5, 3]])
B = np.array([[10], [5]])
C = np.array([[-2], [-3], [-4]])
print(LinearProgram(A, B, C, False).simplexSolver())
