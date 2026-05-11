using ElasticityQMC
using SimpleFiniteElements
import  SimpleFiniteElements.MeshGen: max_elt_diameter
import StaticArrays: SA
using Printf

include("pde_common.jl")

z = rand(s) .- 1/2
@printf("\nRandom K, determinisitic μ (no fft):\n")
xtable1 = zeros(ngrids, ngrids)
@printf("\n%4s  %15s  %8s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ, ∇μ, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand_random_K!(z, α, μ, ∇μ, idx, pstore)
    xtable1[grid,1] = Φ
    h = max_elt_diameter(pstore.dof.mesh)
    @printf("%4d  %15.10f  %8d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
corrections1 = extrapolate!(xtable1, 2, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable1[i,j])
    end
    @printf("\n")
end

rate1 = check_rates(xtable1)
@printf("\nApparent convergence rates:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", rate1[i,j])
    end
    @printf("\n")
end

y = rand(s) .- 1/2

xtable2 = zeros(ngrids, ngrids)
@printf("\nRandom K and μ (with fft):\n")
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
corrections2 = extrapolate!(xtable2, 2, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable2[i,j])
    end
    @printf("\n")
end

rate2 = check_rates(xtable2)
@printf("\nApparent convergence rates:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", rate2[i,j])
    end
    @printf("\n")
end

@printf("\nRandom K and μ (no fft):\n")
@printf("\n%4s  %15s  %5s  %8s\n\n", "grid", "Lₕ", "DoF", "h")
for grid = 1:ngrids
    pstore = PDEStore(mesh[grid], conforming_elements, Λ, μ₀, ∇μ₀, f, solver,
		      pcg_tol, pcg_maxits)
    Φ, num_its = integrand_random_K_μ!(y, z, α, idx, pstore)
    xtable3[grid,1] = Φ
    h = max_elt_diameter(pstore.dof.mesh)
    @printf("%4d  %15.10f  %5d  %8.4f\n", 
            grid, Φ, pstore.dof.num_free, h)
end
corrections3 = extrapolate!(xtable3, 2, 2)
@printf("\nExtrapolated values of L:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", xtable3[i,j])
    end
    @printf("\n")
end

rate3 = check_rates(xtable3)
@printf("\nApparent convergence rates:\n")
for i = 1:ngrids
    for j = 1:i
        @printf("%12.10f  ", rate3[i,j])
    end
    @printf("\n")
end
