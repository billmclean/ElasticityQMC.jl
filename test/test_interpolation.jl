using ElasticityQMC
import ElasticityQMC.InterpolatedCoefs: slow_K, slow_μ, slow_∂₁μ, slow_∂₂μ
import Printf: @printf
using PyPlot

let

α = 2.0
Λ = 2.0
standard_resolution = (256, 256)
high_resolution = (512, 512)
n = 15
idx = double_indices(n)
s = lastindex(idx)

y = rand(s) .- 1/2
z = rand(s) .- 1/2

istore = InterpolationStore(idx, α, standard_resolution, high_resolution)

@printf("Using %d x %d grid to compute interpolants for λ and μ.\n", 
	standard_resolution[1], standard_resolution[2])
@printf("Using %d x %d grid to compute interpolants for components of ∇μ.\n",
	high_resolution[1], high_resolution[2])
K = interpolated_K!(z, istore, Λ)
μ, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore)

M₁ = 3 * high_resolution[1]
M₂ = 3 * high_resolution[2]
@printf("Computing interpolation errors on %d x %d grid (%d threads)\n", 
	M₁, M₂, Threads.nthreads())
x₁ = range(0, 1, length=M₁)
x₂ = range(0, 1, length=M₂)

K_error = 0.0
μ_error = 0.0
∂₁μ_error = 0.0
∂₂μ_error = 0.0
Threads.@threads for j in eachindex(x₂)
    for i in eachindex(x₁)
	K_error = max(K_error, abs(
                      K(x₁[i], x₂[j]) - slow_K(x₁[i], x₂[j], z, α, Λ, idx)))
	μ_error = max(μ_error, abs( 
                      μ(x₁[i], x₂[j]) - slow_μ(x₁[i], x₂[j], y, α, idx)))
	∂₁μ_error = max(∂₁μ_error, abs(
                      ∂₁μ(x₁[i], x₂[j]) - slow_∂₁μ(x₁[i], x₂[j], y, α, idx)))
	∂₂μ_error = max(∂₂μ_error, abs(
                      ∂₂μ(x₁[i], x₂[j]) - slow_∂₂μ(x₁[i], x₂[j], y, α, idx)))
    end
end

@printf("Max interpolations errors for\n")
@printf("\t       K: %0.3e\n", K_error)
@printf("\t       μ: %0.3e\n", μ_error)
@printf("\t     ∂₁μ: %0.3e\n", ∂₁μ_error)
@printf("\t     ∂₂μ: %0.3e\n", ∂₂μ_error)

end # let
