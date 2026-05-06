using ElasticityQMC
using Printf
using JLD2

exno = 2
Λ = 1_000.0
#Λ = 1.0

# Load already computed results
ngrids = 5
QMC_levels = 6
conforming_elements = false
use_fft = false

soln_file = soln_filename(exno, Λ, ngrids, QMC_levels, conforming_elements,
                          use_fft)
@load soln_file choices  L  elapsed  FEM_dof  FEM_h  Nvals
(; ngrids, QMC_levels, conforming_elements, use_fft) = choices

@printf("Example 2 with these choices:\n")
display(choices)

@printf("FEM meshes:\n\n")
@printf("%2s  %8s  %6s\n\n", "j", "h_j", "DoF")
for grid = 1:ngrids
    @printf("%2d  %8.5f  %6d\n", grid, FEM_h[grid], FEM_dof[grid])
end

# Compute expected values of L
EL = zeros(size(L))
for l = 1:QMC_levels, grid = 1:ngrids
    EL[grid,l] = sum(L[grid,l]) / Nvals[l]
end

# Table 3: example of extrapolating the expected values
xtable = zeros(ngrids, ngrids)
l = 4
xtable[:,1] = EL[:,l]
Δ = extrapolate!(xtable, 2)

@printf("\n\nRichardson extrapolation of expected values of L when N = %d\n\n",
        Nvals[l])

for row = 1:ngrids
    @printf("%8.4f & ", FEM_h[row])
    for col = 1:row
        @printf("%13.10f & ", xtable[row,col])
    end
    for col = row+1:ngrids
        @printf("%13s & ", "")
    end
    @printf("\n")
end

@printf("\nExtrapolation corrections:\n\n")
for row = 1:ngrids-1
    for col = 1:row
        @printf("%15.2e ", Δ[row,col])
    end
    @printf("\n")
end

# Table 4: errors and convergence rates
@printf("\nConvergence w.r.t. N:\n\n")
EL, ELx = sum_then_extrapolate(L)
#Lx, ELx = extrapolate_then_sum(L)
err = zeros(QMC_levels)
err[1] = ELx[1] - ELx[end]
@printf("%5d &%12.8f &%10.2e\n", Nvals[1], ELx[1], err[1])
for l = 2:QMC_levels-1
    err[l] = ELx[l] - ELx[end]
    if sign(err[l-1]) == sign(err[l]) 
        rate = log2(err[l-1]/err[l])
        @printf("%5d &%12.8f &%10.2e &%8.3f\n", Nvals[l], ELx[l], err[l], rate)
    else
        @printf("%5d &%12.8f &%10.2e &%8s\n", Nvals[l], ELx[l], err[l], "*")
    end
end
l = QMC_levels
@printf("%5d &%12.8f\n", Nvals[l], ELx[l])

