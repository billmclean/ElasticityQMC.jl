exno = 2
include("compute_solution.jl")
Λ = 1_000.0
ngrids = 4
choices, msg1, msg2, L, L_det, pcg_its, elapsed = read_results(exno, Λ, ngrids)
