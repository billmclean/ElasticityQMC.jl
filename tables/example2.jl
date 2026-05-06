using SimpleFiniteElements
using ElasticityQMC
import StaticArrays: SA
using Printf
using JLD2

choices = (
#           Λ = 1.0,
           Λ = 1_000.0,
           ngrids = 5,
           QMC_levels = 6,
           conforming_elements = false,
           mesh_order = 2,
           solver = :pcg,
           pcg_tol = 1e-10,
           pcg_maxits = 100,
           h_coarse = 0.2,
           n = 22,
           α = 2.0,
           use_fft = false,
           N_std = 256, # used for K and μ
           N_hi  = 512, # used for ∇μ
          )

@printf("Example 2 with choices:\n")
display(choices)

# Finite element method

(; h_coarse, mesh_order, ngrids) = choices

domain_path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(domain_path)
mesh = FEMesh(gmodel, h_coarse, order=mesh_order, save_msh_file=false,
              refinements=ngrids-1, verbosity=2)
FEM_h = zeros(ngrids)
for grid = 1:ngrids
    FEM_h[grid] = max_elt_diameter(mesh[grid])
end

# Interpolation scheme

(; n, α, use_fft, N_std, N_hi) = choices

idx = double_indices(n)
if use_fft
    istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))
end

# QMC parameters

s₁ = 0
s₂ = lastindex(idx)
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s₁+s₂, qmc_path)

# Compute functional values

(; Λ, conforming_elements, solver, pcg_tol, pcg_maxits, QMC_levels) = choices

μ(x₁, x₂) = 1 + x₁ + x₂
∇μ(x₁, x₂) = SA[1.0, 1.0]
f(x, y) = SA[1-y^2, 2x-20.0]

L = Matrix{Vector{Float64}}(undef, ngrids, QMC_levels)
elapsed = zeros(ngrids)
FEM_dof = zeros(Int64, ngrids)

@printf("\nPerforming FEM/QMC calculations ...\n")
for grid = 1:ngrids
    @printf("\t%d. h = %0.4f: ", grid, FEM_h[grid])
    start = time()
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
                      pcg_tol, pcg_maxits)
    FEM_dof[grid] = pstore.dof.num_free
    if use_fft
        for l = 1:QMC_levels
            L[grid,l], _ = simulations_random_K!(pts[l], μ, ∇μ, pstore, istore)
        end
    else
        for l = 1:QMC_levels
            L[grid,l], _ = simulations_random_K!(pts[l], α, μ, ∇μ, idx, pstore)
        end
    end
    elapsed[grid] = time() - start
    @printf("%8.4f seconds\n", elapsed[grid])
end
@printf("\tDone!\n")

exno = 2
save_soln(exno, choices, L, elapsed, FEM_dof, FEM_h, Nvals)
