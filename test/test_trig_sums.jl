import FFTW
import ElasticityQMC.InterpolatedCoefs: sin_sin_sum!, cos_sin_sum!,
					 sin_cos_sum!
using Printf

function S₀(x₁::Float64, x₂::Float64, a::Matrix{Float64})
    Σ = 0.0
    for l = 1:size(a, 2)
	for k = 1:size(a, 1)
	    Σ += a[k,l] * sinpi(k * x₁) * sinpi(l * x₂)
	end
    end
    return Σ
end

function S₁(x₁::Float64, x₂::Float64, a::Matrix{Float64})
    Σ = 0.0
    for l = 1:size(a, 2)
	for k = 1:size(a, 1)
	    Σ += a[k,l] * cospi(k * x₁) * sinpi(l * x₂)
	end
    end
    return Σ
end

function S₂(x₁::Float64, x₂::Float64, a::Matrix{Float64})
    Σ = 0.0
    for l = 1:size(a, 2)
	for k = 1:size(a, 1)
	    Σ += a[k,l] * sinpi(k * x₁) * cospi(l * x₂)
	end
    end
    return Σ
end

N₁, N₂ = 6, 4
a = tril(randn(N₁-1, N₂-1))
n₁, n₂ = size(a)
println("The matrix of coefficients.")
display(a)
x₁ = range(0, 1, length=N₁+1)
x₂ = range(0, 1, length=N₂+1)
slow_Svals = Float64[ S₀(x₁[i], x₂[j], a) for i in eachindex(x₁),
		                          j in eachindex(x₂) ]
fast_Svals = zeros(N₁+1, N₂+1)
sin_sin_sum!(fast_Svals, a)
@printf("Max error computing S₀ = %0.3e\n", 
	maximum(abs, fast_Svals - slow_Svals))

slow_Svals = Float64[ S₁(x₁[i], x₂[j], a) for i in eachindex(x₁),
		                          j in eachindex(x₂) ]
cos_sin_sum!(fast_Svals, a)
@printf("Max error computing S₁ = %0.3e\n", 
	maximum(abs, fast_Svals - slow_Svals))

slow_Svals = Float64[ S₂(x₁[i], x₂[j], a) for i in eachindex(x₁),
		                              j in eachindex(x₂) ]
sin_cos_sum!(fast_Svals, a)
@printf("Max error computing S₂ = %0.3e\n", 
	maximum(abs, fast_Svals - slow_Svals))
