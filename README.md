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

One of the dependencies is an unregistered package.  
After activating `ElasticityQMC` you need to enter the package manager via `]`
and compile the dependencies, one of which is an unregistered:
```
(ElasticityQMC) pkg> add https://github.com/billmclean/SimpleFiniteElements.jl.git
(ElasticityQMC) pkg> instantiate
```

The scripts in `figures/` and `tables/` generate the figures and tables in
Section 6 of the paper.  The actual computation of the linear functionals 
is performed by `example2.jl` and `example3.jl`, run for each choice
of $\Lambda$ thereby creating four data files.  
```
example2_1.0_G7_L6.jld2
example2_1000.0_G7_L6.jld2
example3_1.0_G7_L6.jld2
example3_1000.0_G7_L6.jld2
```
Once these files have been created, `table_3_4.jl` and `table_5.jl` will 
produce the tables in the paper.

Note that `example2.jl` and `example3.jl` call multithreaded functions so
you should invoke julia with the `-t` option set to the number of (performance)
cores in your CPU for faster execution.  On my Ryzen 7 9700X with 32GB of RAM,
and using all eight cores (`julia -t8`), `example2.jl` and `example3.jl` ran 
for about 50 minutes and 85 minutes, respectively.
