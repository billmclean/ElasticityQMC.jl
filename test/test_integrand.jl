using ElasticityQMC
using SimpleFiniteElements
import  SimpleFiniteElements.MeshGen: max_elt_diameter
import StaticArrays: SA
using Printf

include("pde_common.jl")

z = rand(s) .- 1/2
@printf("\nRandom K, determinisitic μ:\n")
xtable1 = zeros(ngrids, ngrids)
@printf("\n%4s  %15s  %5s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand_random_K!(z, μ, ∇μ, pstore, istore)
    xtable1[grid,1] = Φ
    h = max_elt_diameter(pstore.dof.mesh)
    @printf("%4d  %15.10f  %5d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
corrections = extrapolate!(xtable1, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable1[i,j])
    end
    @printf("\n")
end

y = rand(s) .- 1/2

xtable2 = zeros(ngrids, ngrids)
@printf("\nRandom K and μ:\n")
@printf("\n%4s  %15s  %5s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ₀, ∇μ₀, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand_random_K_μ!(y, z, pstore, istore)
    xtable2[grid,1] = Φ
    h = max_elt_diameter(pstore.dof.mesh)
    @printf("%4d  %15.10f  %5d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
corrections = extrapolate!(xtable2, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable2[i,j])
    end
    @printf("\n")
end

