# Copy .npy file contents to a .jld2 file

using PyCall
using JLD2

numpy = pyimport("numpy")
Nvals = [ 2^k for k in 4:11 ]
push!(Nvals, 2^13)
D = Dict{String,Matrix{Float64}}()
jldopen("SPOD_dim256.jld2", "w") do file
    file["Nvals"] = Nvals
    for N in Nvals
        filename = "SPOD_N$(N)_dim256" 
        P = numpy.load(filename * ".npy")
        file[filename] = [ P[j,i] for i = 1:256, j = 1:N ]
    end
end
