# ElasticityQMC

Numerical examples for the paper

> J. Dick, T. Le Gia, W. McLean, K. Mustapha, T. Tran,
> High-order QMC nonconforming FEMs for nearly incompressible planar 
> stochastic elasticity equations

The scripts in `figures/` and `tables/` generate the figures and tables in
Section 6 of the paper.  Note that `table_3.jl`, `table_4.jl` and `table_5.jl`
call (indirectly) a multithreaded function `QMC.simulations!`.  When running
these three scripts, invoke julia with the `-t` option to set the number of 
threads equal to the number of physical cores in your CPU, e.g., for 8-core 
processor do
```
julia -t8
```
