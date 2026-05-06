# ElasticityQMC

Numerical examples for the paper

> J. Dick, T. Le Gia, W. McLean, K. Mustapha, T. Tran,
> High-order QMC nonconforming FEMs for nearly incompressible planar 
> stochastic elasticity equations

After cloning the repo, you will need to do
```
git lfs pull
```
to get the data file `SPOD_dim256.jld2` file from the 
[Git Large File Storage](https://git-lfs.com/) service.  This file holds
the QMC points.  (In addition to `git` itself, the `git-lfs` package must be 
installed on your system.)

The scripts in `figures/` and `tables/` generate the figures and tables in
Section 6 of the paper.  For the actual computation of the linear functionals 
you must call `example2.jl` and `example3.jl` for each choice
of $\Lambda$, thereby creating four `.jld2` data files.  After that,
`table_3_4.jl` and `table_5.jl` will produce the tables in the paper.

Note that `example2.jl` and `example3.jl` call multithreaded functions so
you should invoke julia with the `-t` option set to the number of (performance)
cores in your CPU for faster execution.  On my Intel Core i9-12900 using all
eight performance cores (`julia -t8`), each of these two scripts ran for 
three to four minutes.
