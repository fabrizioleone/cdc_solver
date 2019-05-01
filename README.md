# cdc_solver

An algorithm to divide productivity range according to optimal CDC solution. See `algorithm.pdf` for usage details as well as a description of the algorithm.

The algorithm incorporates the logic of _Arkolakis & Eckert (2017)_, but while theirs solves the CDC problem for a single productivity value, ours solves it for the whole productivity distribution in one run. An implementation of the _A&E_ algorithm in Julia is also provided to run time trials.
