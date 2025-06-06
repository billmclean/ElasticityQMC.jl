using ElasticityQMC
using SimpleFiniteElements
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
    ngrids = 3
    element_description = "non-conforming"
end
mesh = FEMesh(gmodel, hmax, order=mesh_order, save_msh_file=false, 
	      refinements=ngrids-1, verbosity=2)
h_finest = max_elt_diameter(mesh[ngrids])
h_string = @sprintf("%0.4f", h_finest)
N_std = 256
N_hi  = 512

α = 2.0
Λ = 100.0
n = 15
pcg_tol = 1e-10
idx = double_indices(n)
s = lastindex(idx)
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
pstore = PDEStore(mesh; conforming=conforming_elements, solver=solver,
		  pcg_tol=pcg_tol)
istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))

integrand_init!(pstore, Λ, μ, ∇μ, f)
z = rand(s) .- 1/2
Φ, Φ_error, num_its = integrand!(z, Λ, μ, ∇μ, pstore, istore)
@printf("\nRandom K, determinisitic μ:\n")
@printf("\tΦ = %0.6f, Φ_error = %0.2e, num_its = %d, %d, %d\n",
        Φ, Φ_error, num_its[1], num_its[2], num_its[3])

y = rand(s) .- 1/2

integrand_init!(pstore, Λ, f)
Φ, Φ_error, num_its = integrand!(y, z, Λ, pstore, istore)
@printf("\nRandom K and μ:\n")
@printf("\tΦ = %0.6f, Φ_error = %0.2e, num_its = %d, %d, %d\n",
        Φ, Φ_error, num_its[1], num_its[2], num_its[3])

