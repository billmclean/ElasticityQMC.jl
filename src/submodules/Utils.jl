module Utils

import ..Vec64, ..Mat64
using ArgCheck
using LinearAlgebra
using JLD2

# Extending dummy functions from ElasticityQMC.
import ..SPOD_points, ..pcg!, ..extrapolate!, ..extrapolate, ..check_rates

function SPOD_points(s::Integer, path::String)
    D = load(path)
    if s > 256
	error("Dimension s can be at most 256")
    end
    Nvals = D["Nvals"]
    pts = Vector{Mat64}(undef, length(Nvals))
    for k in eachindex(Nvals)
	N = Nvals[k]
        name = "SPOD_N$(N)_dim256"
	pts[k] = D[name][1:s,:] .- 1/2
    end
    return Nvals, pts
end

function pcg!(x::Vector{T}, A::AbstractMatrix{T}, b::Vector{T}, P, tol::T,
        maxits::Integer, wkspace::Matrix{T}) where T <: AbstractFloat
    n = lastindex(x)
    @argcheck size(A) == (n, n)
    @argcheck size(b) == (n,)
    @argcheck size(wkspace) == (n, 4)
    p  = view(wkspace, :, 1)
    r  = view(wkspace, :, 2)
    Ap = view(wkspace, :, 3)
    w  = view(wkspace, :, 4)
    r .= b - A*x
    if norm(r) < tol
        return 0 # zero iterations
    end
    w .= P \ r
    p .= w
    for j = 1:maxits
        mul!(Ap, A, p) # Ap = A * p
        w_dot_r = dot(w, r)
        α = w_dot_r / dot(p, Ap)
        r .-= α * Ap
        x .+= α * p
        if norm(r) < tol
            return j
        end
        w .= P \ r
        β = dot(w, r) / w_dot_r
        p .= w + β * p
    end
    @warn "PCG failed to converge"
    return n
end

"""
    extrapolate!(Φ, rate) -> error_estimate

Applies Richardson extrapolation to the vector `Φ`.

Assumes `Φ[1]` has an error expansion in powers of `N^(-rate)` and
that `Φ[j+1]` is obtained from `Φ[j]` by doubling `N`.
On exit, `Φ[1]` holds the final extrapolated value obtained from
`Φ`, and `error_estimate` holds the final correction that led to
the value of `Φ`.
"""
function extrapolate!(Φ::Vec64, rate::Int64)
    n = lastindex(Φ)
    pow = 2^rate
    correction = 0.0
    for step = 1:n-1
        for j = 1:n-step
            correction = (Φ[j+1] - Φ[j]) / (pow - 1)
            Φ[j] = Φ[j+1] + correction
        end
        pow *= 2^rate
    end
    return correction
end

function extrapolate!(Φ::Matrix{Float64}, rate::Int64)
    m, n = size(Φ)
    error_estimate = Vector{Float64}(undef, m)
    pow = 2^rate
    for step = 1:n-1
        for i = 1:m
            for j = 1:n-step
                correction = (Φ[i,j+1] - Φ[i,j]) / ( pow - 1)
                Φ[i,j] = Φ[i,j+1] + correction
                if step == n-1
                    error_estimate[i] = correction
                end
            end
        end
        pow *= 2^rate
    end
    return error_estimate
end

function extrapolate(Φ::Matrix{Float64}, rate::Int64)
    Φ_extrap = copy(Φ)
    error_estimate = extrapolate!(Φ_extrap, rate)
    return Φ_extrap, error_estimate
end

function check_rates(Φ::Matrix{Float64}, rate::Int64)
    m, n = size(Φ)
    ratio = Matrix{Float64}(undef, m,n-2)
    for j = 1:n-2
        for i = 1:m
            ratio[i,j] = ( Φ[i,j+1] - Φ[i,j] ) / ( Φ[i,j+2] - Φ[i,j+1] )
        end
    end
    return ratio
end

end # module
