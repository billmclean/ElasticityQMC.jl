include("compute_solution.jl")
exno = 2
Λ = 1.0
write_results(exno, Λ, choices)
Λ = 1_000.0
write_results(exno, Λ, choices)

