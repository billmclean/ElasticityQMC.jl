module InterpolatedCoefs

import ..IdxPair, ..InterpolationStore, ..Vec64, ..AVec64, ..Mat64
import FFTW: plan_r2r, RODFT00, REDFT00, r2rFFTWPlan
import Interpolations: cubic_spline_interpolation
import LinearAlgebra.BLAS: scal!
import SpecialFunctions: zeta
using ArgCheck

import ..double_indices, ..interpolated_K!, ..interpolated_μ!

function double_indices(n::Int64)
    idx = Vector{IdxPair}(undef, n*(n+1) ÷ 2)
    j = 1
    for k_plus_l = 2:n+1
        for l = 1:k_plus_l-1
            k = k_plus_l - l
            idx[j] = (k, l)
            j += 1
        end
    end
    return idx
end

function InterpolationStore(idx::Vector{IdxPair}, α::Float64, 
                            standard_resolution::IdxPair,
			    high_resolution::IdxPair)
    M_α = zeta(2α-1) - zeta(2α)
    k_max, l_max = 0, 0
    for j in eachindex(idx)
        k, l = idx[j]
        k_max = max(k, k_max)
        l_max = max(l, l_max)
    end
    N₁, N₂ = standard_resolution
    if k_max > N₁÷2 || l_max > N₂÷2
        error("Interpolation grid is too coarse to resolve Fourier modes")
    end
    coef = zeros(N₁-2, N₂-2)
    vals = zeros(N₁, N₂)
    N₁, N₂ = high_resolution
    ∂₁coef = zeros(N₁-2, N₂-2)
    ∂₁vals = zeros(N₁, N₂)
    ∂₂coef = zeros(N₁-2, N₂-2)
    ∂₂vals = zeros(N₁, N₂)
    plan = plan_r2r(coef, RODFT00)
    return InterpolationStore(idx, α, M_α, coef, vals, ∂₁coef, ∂₁vals,
			      ∂₂coef, ∂₂vals, plan)
end

function interpolated_K!(z::AVec64, istore::InterpolationStore, Λ=1.0)
    (; idx, vals) = istore
    @argcheck length(idx) == size(z, 1)
    KL_expansion!(z, istore)
    if Λ ≠ 1.0
        scal!(Λ, vals)
    end
    N₁, N₂ = size(vals)
    x₁ = range(0, 1, length=N₁)
    x₂ = range(0, 1, length=N₂)
    return cubic_spline_interpolation((x₁, x₂), vals)
end

function interpolated_μ!(y::AVec64, istore::InterpolationStore)
    (; idx, vals, ∂₁vals, ∂₂vals) = istore
    @argcheck length(idx) == size(y, 1)
    KL_expansion_with_gradient!(y, istore)
    N₁, N₂ = size(vals)
    x₁ = range(0, 1, length=N₁)
    x₂ = range(0, 1, length=N₂)
    μ = cubic_spline_interpolation((x₁, x₂), vals)
    N₁, N₂ = size(∂₁vals)
    x₁ = range(0, 1, length=N₁)
    x₂ = range(0, 1, length=N₂)
    ∂₁μ = cubic_spline_interpolation((x₁, x₂), ∂₁vals)
    ∂₂μ = cubic_spline_interpolation((x₁, x₂), ∂₂vals)

    return μ, ∂₁μ, ∂₂μ
end

function KL_expansion!(z::AVec64, istore::InterpolationStore)
    (;idx, α, M_α, coef, vals, plan) = istore
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        coef[k,l] = z[j] * decay_factor
    end
    N₁, N₂ = size(vals)
    sin_sin_sum!(vals, coef, plan)
    for j = 1:N₂, i = 1:N₁
        vals[i,j] = 1 + vals[i,j] / M_α
    end
end

function KL_expansion_with_gradient!(y::AVec64, istore::InterpolationStore)
    (; idx, M_α, coef, ∂₁coef, ∂₁vals, ∂₂coef, ∂₂vals) = istore
    KL_expansion!(y, istore)
    for j in eachindex(idx)
        k, l = idx[j]
        ∂₁coef[k,l] = k * π * coef[k,l] / M_α
        ∂₂coef[k,l] = l * π * coef[k,l] / M_α
    end
    cos_sin_sum!(∂₁vals, ∂₁coef)
    sin_cos_sum!(∂₂vals, ∂₂coef)
end

"""
    sin_sin_sum!(S, a, plan)

Computes the double sin sum

             N₁-1  N₂-1
    S[i, j] =  Σ     Σ    aₖₗ sin(kπx₁) sin(lπx₂)
             k = 1 l = 1

for 

    (x₁,x₂) = (i/N₁, j/N₂), 0 ≤ i ≤ N₁,  0 ≤ j ≤ N₂.
"""
function sin_sin_sum!(S::Matrix{Float64}, a::Matrix{Float64},
                      plan::r2rFFTWPlan)
    S[1,:] .= 0.0
    S[:,1] .= 0.0
    S[end,:] .= 0.0
    S[:,end] .= 0.0
    S[2:end-1, 2:end-1] .= plan * a / 4
end

function sin_sin_sum!(S::Matrix{Float64}, a::Matrix{Float64})
    plan = plan_r2r(a, RODFT00)
    sin_sin_sum!(S, a, plan)
end

"""
    cos_sin_sum!(S, a)

Computes the double sin sum

             N₁-1  N₂-1
    S[i, j] =  Σ     Σ    aₖₗ cos(kπx₁) sin(lπx₂)
             k = 1 l = 1

for 

    (x₁,x₂) = (i/N₁, j/N₂), 0 ≤ i ≤ N₁,  0 ≤ j ≤ N₂.
"""
function cos_sin_sum!(S::Matrix{Float64}, a::Matrix{Float64})
    n₁, n₂ = size(a)
    n₁ += 2 # include end points
    N₁, N₂ = n₁ - 1, n₂ + 1
    if size(S) ≠ (N₁+1, N₂+1)
	throw(DimensionMismatch("Sizes of S and a are incompatible"))
    end
    X = zeros(n₁)
    Y = zeros(n₂)
    cos_plan = plan_r2r(X, REDFT00)
    sin_plan = plan_r2r(Y, RODFT00)
    cos_sin_sum!(S, a, X, cos_plan, Y, sin_plan)
end

function cos_sin_sum!(S::Matrix{Float64}, a::Matrix{Float64},
                      X::Vector{Float64}, cos_plan::r2rFFTWPlan, 
		      Y::Vector{Float64}, sin_plan::r2rFFTWPlan)
    n₁, n₂ = size(a)
    n₁ += 2 # include end points
    N₁, N₂ = n₁ - 1, n₂ + 1
    S[:,1] .= 0.0
    for l = 1:n₂
	X[2:end-1] .= a[:,l]
	S[:,l+1] .= cos_plan * X
    end
    S[:,N₂+1] .= 0.0
    for i = 0:N₁
	Y .= S[i+1,2:end-1]
	S[i+1,2:N₂] .= sin_plan * Y / 4
    end
end

"""
    sin_cos_sum!(S, a)

Computes the double sin sum

             N₁-1  N₂-1
    S[i, j] =  Σ     Σ    aₖₗ sin(kπx₁) cos(lπx₂)
             k = 1 l = 1

for 

    (x₁,x₂) = (i/N₁, j/N₂), 0 ≤ i ≤ N₁,  0 ≤ j ≤ N₂.
"""
function sin_cos_sum!(S::Matrix{Float64}, a::Matrix{Float64})
    n₁, n₂ = size(a)
    n₂ += 2 # include end points
    N₁, N₂ = n₁ + 1, n₂ - 1
    if size(S) ≠ (N₁+1, N₂+1)
	throw(DimensionMismatch("Sizes of S and a are incompatible"))
    end
    X = zeros(n₁)
    Y = zeros(n₂)
    sin_plan = plan_r2r(X, RODFT00)
    cos_plan = plan_r2r(Y, REDFT00)
    sin_cos_sum!(S, a, X, sin_plan, Y, cos_plan)
end

function sin_cos_sum!(S::Matrix{Float64}, a::Matrix{Float64},
                      X::Vector{Float64}, sin_plan::r2rFFTWPlan, 
		      Y::Vector{Float64}, cos_plan::r2rFFTWPlan)
    n₁, n₂ = size(a)
    n₂ += 2 # include end points
    N₁, N₂ = n₁ + 1, n₂ - 1
    S[1,:] .= 0.0
    for k = 1:n₁
	Y[2:end-1] .= a[k,:]
	S[k+1,:] .= cos_plan * Y
    end
    S[N₁+1,:] .= 0.0
    for j = 0:N₂
	X .= S[2:end-1,j+1]
	S[2:N₁,j+1] .= sin_plan * X / 4
    end
end

function K_μ_sums!(y::AVec64, z::AVec64, α::Float64, Λ::Float64,
                   K_coef::Mat64, μ_coef::Mat64, K_vals::Mat64, μ_vals::Mat64, 
                   idx::Vector{IdxPair}, plan::r2rFFTWPlan)
    M_α = zeta(2α-1) - zeta(2α)
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        K_coef[k,l] = y[j] * decay_factor
        μ_coef[k,l] = z[j] * decay_factor
    end
    n₁, n₂ = size(K_coef)
    N₁, N₂ = n₁ + 1, n₂ + 1
    sin_sin_sum!(K_vals, K_coef, plan)
    sin_sin_sum!(μ_vals, μ_coef, plan)
    for j = 0:N₂, i = 0:N₁
        K_vals[i+1,j+1] = Λ * ( 1 + K_vals[i+1,j+1] / M_α )
	μ_vals[i+1,j+1] =       1 + μ_vals[i+1,j+1] / M_α 
    end
    return M_α
end

function slow_K(x₁::Float64, x₂::Float64, y::AVec64, 
                α::Float64, Λ::Float64, idx::Vector{IdxPair})
    M_α = zeta(2α-1) - zeta(2α)
    Σ = 0.0
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        Σ += y[j] * decay_factor * sinpi(k*x₁) * sinpi(l*x₂)
    end
    return Λ * ( 1 + Σ / M_α )
end

function slow_μ(x₁::Float64, x₂::Float64, z::AVec64, 
                α::Float64, idx::Vector{IdxPair})
    M_α = zeta(2α-1) - zeta(2α)
    Σ = 0.0
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        Σ += z[j] * decay_factor * sinpi(k*x₁) * sinpi(l*x₂)
    end
    return  1 + Σ / M_α
end

function slow_∂₁μ(x₁::Float64, x₂::Float64, z::AVec64, 
                  α::Float64, idx::Vector{IdxPair})
    N₁ = length(x₁) - 1
    N₂ = length(x₂) - 1
    M_α = zeta(2α-1) - zeta(2α)
    Σ = 0.0
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        Σ += z[j] * decay_factor * k * π * cospi(k*x₁) * sinpi(l*x₂)
    end
    return  Σ / M_α
end

function slow_∂₂μ(x₁::Float64, x₂::Float64, z::AVec64, 
                  α::Float64, idx::Vector{IdxPair})
    N₁ = length(x₁) - 1
    N₂ = length(x₂) - 1
    M_α = zeta(2α-1) - zeta(2α)
    Σ = 0.0
    for j in eachindex(idx)
        k, l = idx[j]
        decay_factor = 1 / (k + l)^(2α)
        Σ += z[j] * decay_factor * l * π * sinpi(k*x₁) * cospi(l*x₂)
    end
    return  Σ / M_α
end

end # module
