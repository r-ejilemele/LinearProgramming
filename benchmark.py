import time
import numpy as np
from lib import load_linear_program
from linear import *

def benchmark_simplex_two_phase_afiro(numIterations: int):
    A, B, C, types = load_linear_program(
        r"C:\Users\rejil\Documents\GitHub\LinearProgramming\tests\data\numpy\afiro.mps.npz"
    )

    outTime = []
    for i in range(numIterations):
        start = time.perf_counter()
        initResult = LinearProgram(A, B, C, False, types).simplexSolver()
        end = time.perf_counter()
        outTime.append(end-start)
    outTimes = np.array(outTime)
    print(f"Duration Mean: {np.mean(outTimes):.6f} seconds")
    print(f"Duration Median: {np.median(outTimes):.6f} seconds")
    print(f"Duration Standard Deviation: {np.std(outTimes):.6f} seconds")


if __name__ == "__main__":
	benchmark_simplex_two_phase_afiro(20)