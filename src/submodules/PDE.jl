module PDE

using SimpleFiniteElements
import SimpleFiniteElements.FEM: average_field
import LinearAlgebra: BLAS, cholesky

import ..PDEStore, ..InterpolationStore, ..extrapolate!, ..pcg!
import ..Vec64, ..Mat64, ..AVec64, ..SA, ..SparseCholeskyFactor, ..IdxPair
import ..interpolated_K!, ..interpolated_μ!
import ..InterpolatedCoefs: slow_K, slow_μ, slow_∂₁μ, slow_∂₂μ

import ..integrand_init!, ..integrand!, ..slow_integrand!

function PDEStore(mesh::Vector{FEMesh};
        conforming::Bool, solver, pcg_tol=0.0, pcg_maxits=100)
    gD = SA[0.0, 0.0]
    essential_bcs = [("Top", gD), ("Bottom", gD), ("Left", gD), ("Right", gD)]
    ngrids = lastindex(mesh)
    dof = Vector{DegreesOfFreedom}(undef, ngrids)
    if conforming
        El = SimpleFiniteElements.Elasticity
    else
        El = SimpleFiniteElements.NonConformingElasticity
    end
    b_free = Vector{Vec64}(undef, ngrids)
    u_free_det = Vector{Vec64}(undef, ngrids)
    P = Vector{SparseCholeskyFactor}(undef, ngrids)
    wkspace = Vector{Mat64}(undef, ngrids)
    u_free = Vector{Vec64}(undef, ngrids)
    u2h = Vector{Vec64}(undef, ngrids)
    for grid in eachindex(mesh)
        if conforming
            dof[grid] = DegreesOfFreedom(mesh[grid], essential_bcs)
        else
            dof[grid] = DegreesOfFreedom(mesh[grid], essential_bcs, El.ELT_DOF)
        end
    end
    return PDEStore(conforming, solver, dof, b_free, u_free_det,
                    P, wkspace, u_free, u2h, pcg_tol, pcg_maxits)
end


"""
    integrand_init!(pstore, Λ, μ, ∇μ, f)

Prepares `pstore` in the case when `K` is random and `μ` is deterministic.
"""
function integrand_init!(pstore::PDEStore, Λ::Float64, μ::Function,
	                 ∇μ::Function, f::Function)
    if pstore.conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, Λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	K = Λ
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, K),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end
    linear_funcs = Dict("Omega" => (El.∫∫f_dot_v!, f))
    deterministic_solve!(pstore, bilinear_forms, linear_funcs)
end

"""
    integrand_init!(pstore, λ, f)

Prepares `pstore` in the case when `λ` is deterministic and `μ` is random.
"""
function integrand_init!(pstore::PDEStore, λ::Function, f::Function)
    if pstore.conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, 1.0)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	μ_plus_λ(x₁, x₂) = 1 + λ(x₁, x₂)
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
					 (El.∫∫μ_∇u_colon_∇v!, 1.0)])
    end
    linear_funcs = Dict("Omega" => (El.∫∫f_dot_v!, f))
    deterministic_solve!(pstore, bilinear_forms, linear_funcs)
end

"""
    integrand_init!(pstore, Λ, f)

Prepares `pstore` in the case when both `λ` and `μ` are random.
"""
function integrand_init!(pstore::PDEStore, Λ::Float64, f::Function)
    if pstore.conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, Λ),
                                          (El.∫∫2μ_εu_εv!, 1.0)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	μ_plus_λ(x₁, x₂) = 1 + Λ
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
					 (El.∫∫μ_∇u_colon_∇v!, 1.0)])
    end
    linear_funcs = Dict("Omega" => (El.∫∫f_dot_v!, f))
    deterministic_solve!(pstore, bilinear_forms, linear_funcs)
end

function deterministic_solve!(pstore::PDEStore, bilinear_forms::Dict,
                              linear_funcs::Dict)

    (; dof, b_free, u_free_det, P, wkspace, u_free, u2h) = pstore
    ngrids = lastindex(dof)
    Φ_det = Vec64(undef, ngrids)
    # The vector u_free_det is the solution of the deterministic problem, i.e.,
    # for the constant λ and μ obtained when y = z = 0.  For general y and z,
    # we use u_free_det as the first guess for the PCG solver, and precondition 
    # using the Cholesky factor from the deterministic problem.
    for grid in eachindex(dof)
        A_free, _ = assemble_matrix(dof[grid], bilinear_forms, 2)
        num_free, num_fixed = dof[grid].num_free, dof[grid].num_fixed
        b_free[grid], _ = assemble_vector(dof[grid], linear_funcs, 2)
#        u_free_det[grid] = Vec64(undef, 2num_free)
        P[grid] = cholesky(A_free)
        u_free_det[grid] = P[grid] \ b_free[grid]
        wkspace[grid] = Mat64(undef, 2num_free, 4)
        u_free[grid] = Vec64(undef, 2num_free)
        u2h[grid] = zeros(num_free + num_fixed)
        for k = 1:num_free
            u2h[grid][k] = u_free_det[grid][num_free+k]
        end
        Φ_det[grid], _ = average_field(u2h[grid], "Omega", dof[grid])
    end
    Φ_det_error = extrapolate!(Φ_det, 2)
    return Φ_det[1], Φ_det_error
end

function integrand!(z::AVec64, Λ::Float64, μ::Function, ∇μ::Function,
	            pstore::PDEStore, istore::InterpolationStore)
    conforming = pstore.conforming
    K_ = interpolated_K!(z, istore, Λ)
    K = (x₁, x₂) -> K_(x₁, x₂)
    if conforming
        El = SimpleFiniteElements.Elasticity
        λ(x₁, x₂) = K(x₁, x₂) - μ(x₁, x₂)
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, K),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

function integrand!(y::AVec64, K::Function, 
	            pstore::PDEStore, istore::InterpolationStore)
    conforming = pstore.conforming
    μ_, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, λ)
    μ = (x₁, x₂) -> μ_(x₁, x₂)
    if conforming
        λ(x₁, x₂) = K(x₁, x₂) - μ(x₁, x₂)
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
	∇μ(x₁, x₂) = SA[ ∂₁μ(x₁, x₂), ∂₂μ(x₁, x₂) ]
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, K),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

function integrand!(y::AVec64, z::AVec64, Λ::Float64, 
	            pstore::PDEStore, istore::InterpolationStore)
    conforming = pstore.conforming
    K_ = interpolated_K!(z, istore, Λ)
    K = (x₁, x₂) -> K_(x₁, x₂)
    μ_, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore)
    μ = (x₁, x₂) -> μ_(x₁, x₂)
    if conforming
        λ(x₁, x₂) = K(x₁, x₂) - μ(x₁, x₂)
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	∇μ(x₁, x₂) = SA[ ∂₁μ(x₁, x₂), ∂₂μ(x₁, x₂) ]
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, K),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

"""
    slow_integrand!(y, z, α, Λ, idx, pstore)

Computes `K` and `μ` directly, without using FFT and interpolation.
"""
function slow_integrand!(y::AVec64, z::AVec64, α::Float64, Λ::Float64, 
	                 idx::Vector{IdxPair}, pstore::PDEStore)
    conforming = pstore.conforming
    K(x₁, x₂) = slow_K(x₁, x₂, y, α, Λ, idx)
    μ(x₁, x₂) = slow_μ(x₁, x₂, z, α, idx)
    if conforming
        El = SimpleFiniteElements.Elasticity
        λ(x₁, x₂) = K(x₁, x₂) - μ(x₁, x₂)
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	∂₁μ(x₁, x₂) = slow_∂₁μ(x₁, x₂, z, α, idx)
	∂₂μ(x₁, x₂) = slow_∂₂μ(x₁, x₂, z, α, idx)
	∇μ(x₁, x₂) = SA[ ∂₁μ(x₁, x₂), ∂₂μ(x₁, x₂) ]
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, K),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

function random_solve!(pstore::PDEStore, bilinear_forms::Dict)
    (; dof, solver, b_free, u_free_det, P, wkspace, u_free, u2h, 
         pcg_tol, pcg_maxits) = pstore
    ngrids = lastindex(dof)
    Φ = Vector{Float64}(undef, ngrids)
    num_its = zeros(Int64, ngrids)
    for grid in eachindex(dof)
        A_free, _ = assemble_matrix(dof[grid], bilinear_forms, 2)
        if solver == :direct
            u_free[grid] .= A_free \ b_free[grid]
        elseif solver == :pcg
            u_free[grid] .= u_free_det[grid] # Starting vector
	    num_its[grid] = pcg!(u_free[grid], A_free, b_free[grid], P[grid], 
                           pcg_tol, pcg_maxits, wkspace[grid])
        else
            error("Unknown solver.")
        end
        num_free = dof[grid].num_free
        for k = 1:num_free
            u2h[grid][k] = u_free[grid][num_free+k]
        end
        Φ[grid], _ = average_field(u2h[grid], "Omega", dof[grid])
    end
    Φ_error = extrapolate!(Φ, 2)
    return Φ[1], Φ_error, num_its
end

end 
