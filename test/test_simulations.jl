include("pde_common.jl")

@printf("\nRandom K, determinisitic μ:\n")

grid = 2
s₁ = 0
s₂ = lastindex(idx)
n = 22
idx = double_indices(n)
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s₁+s₂, qmc_path)

pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
                  pcg_tol, pcg_maxits)
L1, pcg_its1 = simulations_random_K!(pts[4], μ, ∇μ, pstore, istore)
EL1 = sum(L1) / length(L1)
@printf("Expected value of functional = %0.4f\n", EL1)
num_its1 = round(Int64, sum(pcg_its1) / length(pcg_its1))
@printf("Average number of CG iterations = %d\n", num_its1)

@printf("\nRandom K and μ:\n")
n = 15
idx = double_indices(n)
s₁ = s₂ = lastindex(idx)
Nvals, pts = SPOD_points(s₁+s₂, qmc_path)

pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ₀, ∇μ₀, f, solver,
                  pcg_tol, pcg_maxits)
L2, pcg_its2 = simulations_random_K_μ!(pts[4], pstore, istore)
EL2 = sum(L2) / length(L2)
@printf("Expected value of functional = %0.4f\n", EL2)
num_its2 = round(Int64, sum(pcg_its2) / length(pcg_its2))
@printf("Average number of CG iterations = %d\n", num_its2)
