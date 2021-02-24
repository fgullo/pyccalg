# GeneralWeightedCC
Algorithms for `Correlation Clustering` in general weighted graphs.
Specifically, an implementation of the (linear-programming + region-growing) O(log n)-approximation algorithm by Demaine et al., TCS 2006 is provided (see [here](https://www.sciencedirect.com/science/article/pii/S0304397506003227) for a description of the algorithm).


Requirements:

* [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) and/or [PuLP](https://pypi.org/project/PuLP/)
* However, with minimal adaptation, any other linear-programming `Python` library can alternatively be used

Usage:

``` python ./src/ologncc.py -d <DATASET_FILE>```

* Optional parameter `-r <LB>,<UB>`, if you want to generate random edge weights from `[LB,UB]` range 

