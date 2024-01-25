exno = 4
include("common_inputs.jl")
L_error_1,    pcg_its_1    = create_tables(exno; Λ=1.0, nrows=5)
L_error_1000, pcg_its_1000 = create_tables(exno; Λ=1000.0, nrows=5)
