using ElasticityQMC
import ElasticityQMC.InterpolatedCoefs: interpolated_λ!, interpolated_μ!,
                                         slow_λ, slow_μ, slow_∂₁μ, slow_∂₂μ
import Printf: @printf
using PyPlot

let

α = 2.0
Λ = 2.0
standard_resolution = (256, 256)
high_resolution = (512, 512)
n = 22 
idx = double_indices(n)
s = lastindex(idx)
@printf("KL expansions have %d terms.\n", s)

y = rand(s) .- 1/2
z = rand(s) .- 1/2

istore = InterpolationStore(idx, α, standard_resolution, high_resolution)

@printf("Using %d x %d grid to computing interpolants for λ and μ.\n", 
	standard_resolution[1], standard_resolution[2])
@printf("Using %d x %d grid to computing interpolants for components of ∇μ.\n",
	high_resolution[1], high_resolution[2])
M₁ = 3 * high_resolution[1]
M₂ = 3 * high_resolution[2]
@printf("Evaluating interpolants on %d x %d grid\n", M₁, M₂)

x₁ = range(0, 1, length=M₁)
x₂ = range(0, 1, length=M₂)

start = time()
λ = interpolated_λ!(z, istore, Λ)
μ, μ_plus_λ, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, (x₁, x₂) -> Λ)

λ_interp = zeros(M₁, M₂)
μ_interp = zeros(M₁, M₂)
μ_plus_λ_interp = zeros(M₁, M₂)
∂₁μ_interp = zeros(M₁, M₂)
∂₂μ_interp = zeros(M₁, M₂)
for j in eachindex(x₂)
    for i in eachindex(x₂)
	λ_interp[i,j] = λ(x₁[i], x₂[j])
	μ_interp[i,j] = μ(x₁[i], x₂[j])
	μ_plus_λ_interp[i,j] = μ_plus_λ(x₁[i], x₂[j])
	∂₁μ_interp[i,j] = ∂₁μ(x₁[i], x₂[j])
	∂₂μ_interp[i,j] = ∂₂μ(x₁[i], x₂[j])
    end
end
elapsed = time() - start
@printf("Total time (FFT + interpolation) %d seconds.\n", elapsed)

figure(1)
x₁ = range(0, 1, length=50)
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

figure(2, figsize=(6.4,9.6))
x₁ = range(0, 1, length=40)
x₂ = range(0, 1, length=50)
subplot(2, 1, 1)
contour(x₁, x₂, ∂₁μ.(x₁', x₂), levels=10)
colorbar()
title("∂₁μ")
subplot(2, 1, 2)
contour(x₁, x₂, ∂₂μ.(x₁', x₂), levels=10)
colorbar()
title("∂₂μ")

@printf("Now evaluate sums directly on the same %d x %d grid ...", M₁, M₂)

x₁ = range(0, 1, length=M₁)
x₂ = range(0, 1, length=M₂)

start = time()

λ_val = zeros(M₁, M₂)
μ_val = zeros(M₁, M₂)
μ_plus_λ_val = zeros(M₁, M₂)
∂₁μ_val = zeros(M₁, M₂)
∂₂μ_val = zeros(M₁, M₂)
for j in eachindex(x₂)
    for i in eachindex(x₂)
	λ_val[i,j] = slow_λ(x₁[i], x₂[j], z, α, Λ, idx) 
	μ_val[i,j] = slow_μ(x₁[i], x₂[j], y, α, Λ, idx) 
	μ_plus_λ_val[i,j] = slow_μ(x₁[i], x₂[j], y, α, Λ, idx) - Λ 
	∂₁μ_val[i,j] = slow_∂₁μ(x₁[i], x₂[j], y, α, Λ, idx) 
	∂₂μ_val[i,j] = slow_∂₂μ(x₁[i], x₂[j], y, α, Λ, idx) 
    end
end
elapsed = time() - start

@printf(" in %d seconds.\n", elapsed)

end # let
