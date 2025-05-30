using ElasticityQMC
using LinearAlgebra
using SimpleFiniteElements
import SimpleFiniteElements.Elasticity: ∫∫f_dot_v!, ∫∫λ_div_u_div_v!, ∫∫2μ_εu_εv!
using StaticArrays
using PyPlot

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
Λ = 2.0
λ0 = Λ
μ0 = 1.0

bilinear_forms = Dict("Omega" => [(∫∫λ_div_u_div_v!, λ0),
				  (∫∫2μ_εu_εv!, μ0)])
f(x, y) = SA[1-y^2, 2x-20.0]
linear_funcs = Dict("Omega" => (∫∫f_dot_v!, f))
gD = SA[0.0, 0.0]
essential_bcs = [("Top", gD), ("Bottom", gD), ("Left", gD), ("Right", gD)]

hmax = 0.025
mesh = FEMesh(gmodel, hmax, save_msh_file=false, verbosity=2)
dof = DegreesOfFreedom(mesh, essential_bcs)

A_free_det, _ = assemble_matrix(dof, bilinear_forms, 2)
b_free, _ = assemble_vector(dof, linear_funcs, 2)
P = cholesky(A_free_det)
u_free_det = P \ b_free

n = 15
α = 2.0
idx = double_indices(n)
s₁ = lastindex(idx)
z = rand(s₁) .- 1/2
N₁ = N₂ = 255
standard_resolution = (256, 256)
high_resolution = (512, 512)
istore = InterpolationStore(idx, α, standard_resolution, high_resolution)
K = interpolated_K!(z, istore, Λ)
λ(x₁, x₂) = K(x₁, x₂) - μ0
bilinear_forms = Dict("Omega" => [(∫∫λ_div_u_div_v!, λ),
				  (∫∫2μ_εu_εv!, μ0)])
A_free, _ = assemble_matrix(dof, bilinear_forms, 2)

tol = 1e-7
wkspace = zeros(2dof.num_free, 4)
u_free = copy(u_free_det) # starting vector
unknowns = length(u_free)
println("Solving $unknowns x $unknowns linear system.")
maxits = 200
num_its = pcg!(u_free, A_free, b_free, I, tol, maxits, wkspace)
println("Required $num_its PCG iterations with no preconditioning")
u_free = copy(u_free_det) # reset 
num_its = pcg!(u_free, A_free, b_free, P, tol, maxits, wkspace)
println("Required $num_its PCG iterations with preconditioning")

figure(1)
x1 = range(0, 1, length=150)
x2 = range(0, 1, length=150)
z = Float64[ λ(x1[i1], x2[i2]) for i2 = 1:150, i1 = 1:150 ]
contourf(x1, x2, z)
colorbar()
title("The Coefficient λ")
