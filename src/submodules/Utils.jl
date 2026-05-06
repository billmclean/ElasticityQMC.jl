module Utils

import ..SPOD_points, ..pcg!, ..extrapolate!, ..extrapolate, ..check_rates,
       ..soln_filename, ..save_soln, 
       ..sum_then_extrapolate, ..extrapolate_then_sum
import ..Vec64, ..Mat64
import ArgCheck: @argcheck
import LinearAlgebra: dot, norm
import JLD2: load, jldsave

"""
    SPOD_points(s, path)

Reads QMC points of dimension `s` from `path` (in our case the file
`qmc_points/SPOD_dim256.jld2`).
"""
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

"""
    pcg!(x, A, b, P, tol, maxits, wkspace) 

Preconditioned conjugate gradient solver for an `n×n` linear system `Ax=b`.  
The iteration stops when the 2-norm of the relative residual is smaller than 
`tol`, or when the number of iterations exceeds `maxits`.  The `wkspace`
array must be `n×4`.
"""
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
    tol_b = tol * norm(b)
    if norm(r) < tol_b
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
        if norm(r) < tol_b
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

function extrapolate(Φ::Matrix{Float64}, rate::Int64)
    Φ_extrap = copy(Φ)
    error_estimate = extrapolate!(Φ_extrap, rate)
    return Φ_extrap, error_estimate
end

"""
    extrapolate!(xtable, rate)

Saves the complete extrapolation table.
"""
function extrapolate!(xtable::Matrix{Float64}, rate::Int64)
    m, n = size(xtable)
    correction = zeros(m-1,n-1)
    pow = 2^rate
    for j = 2:n
        for i = j:m
            correction[i-1,j-1] = (xtable[i,j-1] - xtable[i-1,j-1]) / ( pow - 1)
            xtable[i,j] = xtable[i,j-1] + correction[i-1,j-1]
        end
        pow *= 2^rate
    end
    return correction
end

"""
    check_rates(xtable)

Computes the empirical convergence rates for the columns of the extrapolation
table.
"""
function check_rates(xtable::Matrix{Float64})
    m, n = size(xtable)
    rate = zeros(m, n)
    for j = 1:n
        for i = j+1:m-1
            rate[i,j] = log2(( xtable[i,j] - xtable[i-1,j] ) 
                             / ( xtable[i+1,j] - xtable[i,j] ))
        end
    end
    return rate
end

"""
    soln_filename(exno, Λ, ngrids, QMC_levels, conforming_elements, use_fft)

Returns the filename for the data file corresponding to the argument list.
"""
function soln_filename(exno::Int64, Λ::Float64, ngrids::Int64, 
                       QMC_levels::Int64, conforming_elements::Bool,
                       use_fft::Bool)
    s = ""
    if conforming_elements
        s = s + "_c"
    end
    if use_fft
        s = s + "_f"
    end
    return "example$(exno)_$(Λ)_G$(ngrids)_L$(QMC_levels)$s.jld2"
end

"""
    save_soln(exno, choices, L, elapsed, FEM_dof, FEM_h, Nvals)

Saves the linear functional values and related quantities to a `.jld2` data 
file.
"""
function save_soln(exno::Int64, choices, L::Matrix{Vector{Float64}}, 
        elapsed::Vector{Float64}, FEM_dof::Vector{Int64}, 
        FEM_h::Vector{Float64}, Nvals::Vector{Int64})
    (; Λ, ngrids, QMC_levels, conforming_elements, use_fft) = choices
    soln_file = soln_filename(exno, Λ, ngrids, QMC_levels, conforming_elements,
                              use_fft)
    jldsave(soln_file; choices, L, elapsed, FEM_dof, FEM_h, Nvals)
end

"""
    sum_then_extrapolate(L)

Sumes to obtain the expected values of the functional for each mesh, and
then extrapolates.
"""
function sum_then_extrapolate(L::Matrix{Vector{Float64}})
    # First sum to compute expected values of L
    EL = zeros(size(L))
    ngrids = size(L, 1)
    QMC_levels = size(L, 2)
    for l = 1:QMC_levels, grid = 1:ngrids
        N = length(L[grid,l])
        EL[grid,l] = sum(L[grid,l]) / N
    end
    # Then extrapolate
    ELx = zeros(QMC_levels)
    xtable = zeros(ngrids, ngrids)
    for l in eachindex(ELx)
        xtable[:,1] = EL[:,l]
        extrapolate!(xtable, 2)
        ELx[l] = xtable[ngrids, ngrids]
    end
    return EL, ELx
end

"""
    extrapolate_then_sum(L)

Extrapolates the functional values for each QMC points, and then sums to
obtain the expected values. Should give the same result `ELx` as
`sum_then_extrapolate`.
"""
function extrapolate_then_sum(L::Matrix{Vector{Float64}})
    #  First extrapolate for each
    ngrids = size(L, 1)
    QMC_levels = size(L, 2)
    Lx = Vector{Vector{Float64}}(undef, QMC_levels)
    xtable = zeros(ngrids, ngrids)
    for l = 1:QMC_levels
        N = length(L[1,l])
        Lx[l] = zeros(N)
        for k = 1:N
            for grid = 1:ngrids
                xtable[grid,1] = L[grid,l][k]
                extrapolate!(xtable, 2)
                Lx[l][k] = xtable[ngrids, ngrids]
            end
        end
    end
    # Then sum to compute expected values of Lx
    ELx = zeros(QMC_levels)
    for l in eachindex(ELx)
        N = length(Lx[l])
        ELx[l] = sum(Lx[l]) / N
    end
    return Lx, ELx
end

end # module
