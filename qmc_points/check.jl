using PyCall
using JLD2

numpy = pyimport("numpy")

D = load("SPOD_dim256.jld2")
println("Computing maximum absolute differences\n")
for (filename, P) in D
    P_numpy = numpy.load(filename * ".npy")
    δ = maximum(abs, P' - P_numpy)
    println(filename, ": ", δ)
end
