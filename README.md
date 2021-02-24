# GeneralWeightedCC
Algorithms for `Correlation Clustering` in general weighted graphs.
Specifically, an implementation of the (linear-programming + region-growing) O(log n)-approximation algorithm by Demaine et al., TCS 2006 is provided (see [here](https://www.sciencedirect.com/science/article/pii/S0304397506003227) for a description of the algorithm).


### Requirements:

* [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) and/or [PuLP](https://pypi.org/project/PuLP/) (`PuLP` comes with two solvers by default: [CBC](https://projects.coin-or.org/Cbc) (linear and integer programming) and [CHOCO](https://choco-solver.org/) (constraint programming), but it can connect to many others (e.g., `GUROBI`, `CPLEX`, `SCIP`, `MIPCL`, `XPRESS`, `GLPK9`) if you have them installed)
* However, with minimal adaptation, any other linear-programming `Python` library can alternatively be used

### Usage:

``` python src/ologncc.py -d <DATASET_FILE> [-r <LB,UB>] [-s {'scipy','pulp'}```

* Optional arguments: 
   * `-r <LB,UB>`, if you want to generate random edge weights from `[LB,UB]` range
   * `-s {'scipy','pulp'}`, to select the solver to be used (default: `'pulp'` (because it is faster))
* Dataset-file format:
   * First line: `#VERTICES \t #EDGES`
   * One line per edge; every line is a quadruple: `NODE1 \t NODE2 \t POSITIVE_WEIGHT \t NEGATIVE_WEIGHT` (`POSITIVE_WEIGHT` and `NEGATIVE_WEIGHT` are ignored if code is run with `-r` option)
   * Look at `data` folder for some examples

