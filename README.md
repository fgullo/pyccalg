# GeneralWeightedCC
Algorithms for `Correlation Clustering` in general weighted graphs.
Specifically, an implementation of the (linear-programming + region-growing) O(log n)-approximation algorithm by Demaine et al., TCS 2006 is provided (see [here](https://www.sciencedirect.com/science/article/pii/S0304397506003227) for a description of the algorithm).


### Requirements:

* [`SciPy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)`v1.6` (or higher) and/or [`PuLP`](https://pypi.org/project/PuLP/)
* `SciPy linprog` comes with various solvers: '*Method `highs-ds` is a wrapper of the C++ high performance dual revised simplex implementation (HSOL). Method `highs-ipm` is a wrapper of a C++ implementation of an interior-point method; it features a crossover routine, so it is as accurate as a simplex solver. Method `highs` chooses between the two automatically. For new code involving linprog, we recommend explicitly choosing one of these three method values instead of `interior-point` (default), `revised simplex`, and `simplex` (legacy)*'. See [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) for more details.
* `PuLP` comes with two solvers by default: [CBC](https://projects.coin-or.org/Cbc) (linear and integer programming) and [CHOCO](https://choco-solver.org/) (constraint programming), but it can connect to many others (e.g., `GUROBI`, `CPLEX`, `SCIP`, `MIPCL`, `XPRESS`, `GLPK9`) if you have them installed
* However, any linear-programming `Python`  (other than `SciPy linprog` or `PuLP`) library can alternatively be used with minimal adaption

### Usage:

``` python src/ologncc.py -d <DATASET_FILE> [-r <LB,UB>] [-s {'scipy','pulp'}]```

* Optional arguments: 
   * `-r <LB,UB>`, if you want to generate random edge weights from `[LB,UB]` range
   * `-s {'scipy','pulp'}`, to select the solver to be used (default: `'pulp'` (it seems faster))
* Dataset-file format:
   * First line: `#VERTICES \t #EDGES`
   * One line per edge; every line is a quadruple: `NODE1 \t NODE2 \t POSITIVE_WEIGHT \t NEGATIVE_WEIGHT` (`POSITIVE_WEIGHT` and `NEGATIVE_WEIGHT` are ignored if code is run with `-r` option)
   * Look at `data` folder for some examples

