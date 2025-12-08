# 18.335 Final Project

The files that I'm submitting for this final project are `revisedSimplex.ipynb`, `lib.py`, and `environment.yml`.

## `revisedSimplex.ipynb`
This Jupyter contains the main code for this final project. It contains two main implementations of the Simplex algorith: one with a Sherman-Morrison update and one with a Forrest-Tomlin update. It also contains code to generate the graphs used in the final paper

## `lib.py`
This file contains functions that parse mps files into matrices to run the simplex algorithm on. It also saves those mps files as `.npz` files that can also be parsed to get a linear program's matrices. 

In order to use this function, you can select a linear program from this [github repo](https://github.com/ozy4dm/lp-data-netlib/tree/main) that contains mps files that are documented on this [netlib](https://www.netlib.org/lp/data/readme) page.

The external dependencies for this project are `matplotlib`, `scipy`, `numpy`, and `pulp`.
