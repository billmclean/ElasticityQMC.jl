# ElasticityQMC

Numerical examples for a paper on QMC for uncertainty quantification of 
elasticity equations.

The scripts in `figures/` and `tables/` generate the figures and tables in
Section 6 of the paper.  Note that `table_3.jl` and `table_4.jl` call
(indirectly) a multithreaded function `QMC.simulations!`.  When running
these two scripts, invoke julia with the `-t` option to set the number of 
threads equal to the number of physical cores in your CPU, e.g., for 8-core 
processor do
```
julia -t8
```
