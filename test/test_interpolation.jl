using Elasticity_QMC
import Elasticity_QMC.InterpolatedCoefs: interpolated_λ!, interpolated_μ!,
                                         slow_λ, slow_μ, slow_∂₁μ, slow_∂₂μ
import Printf: @printf
using PyPlot

α = 2.0
Λ = 2.0
N₁ = 255
N₂ = 127
@printf("Using %d x %d grid to construct interpolant\n", N₁, N₂)
n = 15
idx = double_indices(n)
s = lastindex(idx)

z = rand(s) .- 1/2

istore = InterpolationStore(idx, α, N₁, N₂)

λ = interpolated_λ!(z, istore, Λ)

M₁ = 489
M₂ = 313
@printf("Computing interpolation errors on %d x %d grid\n", M₁, M₂)
x₁ = range(0, 1, length=M₁)
x₂ = range(0, 1, length=M₂)

@printf("Testing λ interpolant.\n")
λ_error = Float64[ λ(x₁[i], x₂[j]) - slow_λ(x₁[i], x₂[j], z, α, Λ, idx) 
                   for i in eachindex(x₁), j in eachindex(x₂) ]
@printf("\tMax interpolation error for λ = %0.3e\n", maximum(abs, λ_error))

@printf("\nTesting μ interpolants.\n")
y = rand(s) .- 1/2
μ, μ_plus_λ, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, (x₁, x₂) -> Λ)
μ_error = Float64[ μ(x₁[i], x₂[j]) - slow_μ(x₁[i], x₂[j], y, α, Λ, idx) 
                   for i in eachindex(x₁), j in eachindex(x₂) ]

@printf("\tMax interpolation error for μ = %0.3e\n", maximum(abs, μ_error))

μ_plus_λ_error = Float64[ ( μ_plus_λ(x₁[i], x₂[j]) 
			   - slow_μ(x₁[i], x₂[j], y, α, Λ, idx) - Λ )
                          for i in eachindex(x₁), j in eachindex(x₂) ]
@printf("\tMax interpolation error for μ_plus_λ = %0.3e\n", 
        maximum(abs, μ_plus_λ_error))
∂₁μ_error = Float64[ ∂₁μ(x₁[i], x₂[j]) - slow_∂₁μ(x₁[i], x₂[j], y, α, Λ, idx) 
                   for i in eachindex(x₁), j in eachindex(x₂) ]
@printf("\tMax interpolation error for ∂₁μ = %0.3e\n", maximum(abs, ∂₁μ_error))
∂₂μ_error = Float64[ ∂₂μ(x₁[i], x₂[j]) - slow_∂₂μ(x₁[i], x₂[j], y, α, Λ, idx) 
                   for i in eachindex(x₁), j in eachindex(x₂) ]
@printf("\tMax interpolation error for ∂₂μ = %0.3e\n", maximum(abs, ∂₂μ_error))

figure(1)
x₁ = range(0, 1, length=40)
x₂ = range(0, 1, length=50)
contour(x₁, x₂, μ.(x₁', x₂), levels=10)
colorbar()
x₁ = range(0, 1, length=20)
x₂ = range(0, 1, length=20)
X₁ = Float64[ x₁[i] for i in eachindex(x₁), j in eachindex(x₂) ]
X₂ = Float64[ x₂[j] for i in eachindex(x₁), j in eachindex(x₂) ]
V₁ = ∂₁μ.(x₁, x₂')
V₂ = ∂₂μ.(x₁, x₂')
quiver(X₁, X₂, V₁, V₂, color="k")
title("The coefficient μ and its gradient")
