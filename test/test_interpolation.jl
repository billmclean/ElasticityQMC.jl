using ElasticityQMC
import ElasticityQMC.InterpolatedCoefs: slow_λ, slow_μ, slow_∂₁μ, slow_∂₂μ
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

@printf("Using %d x %d grid to computing interpolants for λ and μ.\n", 
	standard_resolution[1], standard_resolution[2])
@printf("Using %d x %d grid to computing interpolants for components of ∇μ.\n",
	high_resolution[1], high_resolution[2])
λ = interpolated_λ!(z, istore, Λ)
μ, μ_plus_λ, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, (x₁, x₂) -> Λ)

M₁ = 3 * high_resolution[1]
M₂ = 3 * high_resolution[2]
@printf("Computing interpolation errors on %d x %d grid (%d threads)\n", 
	M₁, M₂, Threads.nthreads())
x₁ = range(0, 1, length=M₁)
x₂ = range(0, 1, length=M₂)

λ_error = zeros(M₁, M₂)
μ_error = zeros(M₁, M₂)
μ_plus_λ_error = zeros(M₁, M₂)
∂₁μ_error = zeros(M₁, M₂)
∂₂μ_error = zeros(M₁, M₂)
Threads.@threads for j in eachindex(x₂)
    for i in eachindex(x₂)
	λ_error[i,j] = λ(x₁[i], x₂[j]) - slow_λ(x₁[i], x₂[j], z, α, Λ, idx) 
	μ_error[i,j] =  μ(x₁[i], x₂[j]) - slow_μ(x₁[i], x₂[j], y, α, Λ, idx) 
	μ_plus_λ_error[i,j] = ( μ_plus_λ(x₁[i], x₂[j]) 
		              - slow_μ(x₁[i], x₂[j], y, α, Λ, idx) - Λ ) 
	∂₁μ_error[i,j] = ∂₁μ(x₁[i], x₂[j]) - slow_∂₁μ(x₁[i], x₂[j], 
						      y, α, Λ, idx) 
	∂₂μ_error[i,j] = ∂₂μ(x₁[i], x₂[j]) - slow_∂₂μ(x₁[i], x₂[j], 
						      y, α, Λ, idx) 
    end
end

@printf("Max interpolations errors for\n")
@printf("\t       λ: %0.3e\n", maximum(abs, λ_error))
@printf("\t       μ: %0.3e\n", maximum(abs, μ_error))
@printf("\tμ_plus_λ: = %0.3e\n", maximum(abs, μ_plus_λ_error))
@printf("\t     ∂₁μ: %0.3e\n", maximum(abs, ∂₁μ_error))
@printf("\t     ∂₂μ: %0.3e\n", maximum(abs, ∂₂μ_error))

end # let
