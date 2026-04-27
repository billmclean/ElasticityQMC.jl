using ElasticityQMC
using SimpleFiniteElements
import  SimpleFiniteElements.MeshGen: max_elt_diameter
import StaticArrays: SA
using Printf

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
hmax = 0.2
conforming_elements = false
solver = :pcg  # :direct or :pcg

if conforming_elements
    mesh_order = 1
    ngrids = 4
    element_description = "conforming"
else
    mesh_order = 2
    ngrids = 5
    element_description = "non-conforming"
end
mesh = FEMesh(gmodel, hmax, order=mesh_order, save_msh_file=false, 
	      refinements=ngrids-1, verbosity=2)
h_finest = max_elt_diameter(mesh[end])
h_string = @sprintf("%0.4f", h_finest)
N_std = 256
N_hi  = 512

α = 2.0
Λ = 100.0
n = 15
pcg_tol = 1e-10
pcg_maxits = 100
idx = double_indices(n)
s = lastindex(idx)
istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))
msg = """
Solving BVP with $element_description finite elements.
Solver is $solver, tol = $pcg_tol.
Finest mesh has h = $h_string.
Using s = $s with α = $α, Λ = $Λ.
Interpolating K and μ using $N_std x $N_std spatial grid."""
println(msg)

μ(x₁, x₂) = 1 + x₁ + x₂
∇μ(x₁, x₂) = SA[1.0, 1.0]
f(x, y) = SA[1-y^2, 2x-20.0]

z = rand(s) .- 1/2
@printf("\nRandom K, determinisitic μ:\n")
xtable = zeros(ngrids, ngrids)
@printf("\n%4s  %15s  %5s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand!(z, Λ, μ, ∇μ, pstore, istore)
    xtable[grid,1] = Φ
    h = max_elt_diameter(pstore.dof.mesh)
    @printf("%4d  %15.10f  %5d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
corrections = extrapolate!(xtable, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable[i,j])
    end
    @printf("\n")
end

y = rand(s) .- 1/2

integrand_init!(pstore, Λ, f)
Φ, num_its = integrand!(y, z, Λ, pstore, istore)
@printf("\nRandom K and μ:\n")
@printf("\n%4s  %15s  %5s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand!(y, z, Λ, μ, ∇μ, pstore, istore)
    h = max_elt_diameter(pstore.dof[grid].mesh)
    @printf("%4d  %15.10f  %5d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
xtable = zeros(ngrids, ngrids)
xtable[:,1] = Φ
corrections = extrapolate!(xtable, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable[i,j])
    end
    @printf("\n")
end

