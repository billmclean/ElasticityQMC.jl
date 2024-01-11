module PDE

using SimpleFiniteElements
import SimpleFiniteElements.FEM: average_field
import LinearAlgebra: BLAS, cholesky

import ..PDEStore, ..InterpolationStore, 
       ..Vec64, ..Mat64, ..AVec64, ..SA, ..SparseCholeskyFactor
import ..InterpolatedCoefs: @unpack_InterpolationStore,
			    interpolated_λ!, interpolated_μ!
import ..Utils: extrapolate!, pcg!

macro unpack_PDEStore(q)
    code =  Expr(:block, [ :($field = $q.$field)
                          for field in fieldnames(PDEStore) ]...)
    esc(code)
end

function PDEStore(mesh::Vector{FEMesh};
        conforming::Bool, solver=:direct, pcg_tol=0.0, pcg_maxits=100)
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
    integrand_init!(pstore, Λ, μ, ∇μ)

Prepares `pstore` in the case when `λ` is random and `μ` is deterministic.
"""
function integrand_init!(pstore::PDEStore, Λ::Float64, μ::Function,
	                 ∇μ::Function, f::Function)
    if pstore.conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, Λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
	μ_plus_λ(x₁, x₂) = μ(x₁, x₂) + Λ
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end
    linear_funcs = Dict("Omega" => (El.∫∫f_dot_v!, f))
    deterministic_solve!(pstore, bilinear_forms, linear_funcs)
end

"""
    integrand_init!(pstore, λ)

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

function deterministic_solve!(pstore::PDEStore, bilinear_forms::Dict,
                              linear_funcs::Dict)
    @unpack_PDEStore(pstore)
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
    @unpack_PDEStore(pstore)
    λ_ = interpolated_λ!(z, istore, Λ)
    λ = (x₁, x₂) -> λ_(x₁, x₂)
    if conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
	μ_plus_λ(x₁, x₂) = λ(x₁, x₂) + μ(x₁, x₂)
        El = SimpleFiniteElements.NonConformingElasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

function integrand!(y::AVec64, λ::Function, 
	            pstore::PDEStore, istore::InterpolationStore)
    @unpack_PDEStore(pstore)
    μ_, μ_plus_λ_, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, λ)
    μ = (x₁, x₂) -> μ_(x₁, x₂)
    if conforming
        El = SimpleFiniteElements.Elasticity
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, λ),
                                          (El.∫∫2μ_εu_εv!, μ)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
        μ_plus_λ = (x₁, x₂) -> μ_plus_λ_(x₁, x₂)
	∇μ(x₁, x₂) = SA[ ∂₁μ(x₁, x₂), ∂₂μ(x₁, x₂) ]
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
                                          (El.∫∫μ_∇u_colon_∇v!, μ),
                                          (El.correction!, ∇μ)])
    end        
    random_solve!(pstore, bilinear_forms)
end

function random_solve!(pstore::PDEStore, bilinear_forms::Dict)
    @unpack_PDEStore(pstore)
    ngrids = lastindex(dof)
    Φ = Vector{Float64}(undef, ngrids)
    for grid in eachindex(dof)
        A_free, _ = assemble_matrix(dof[grid], bilinear_forms, 2)
        if solver == :direct
            u_free[grid] .= A_free \ b_free[grid]
        elseif solver == :pcg
            u_free[grid] .= u_free_det[grid] # Starting vector
            num_its = pcg!(u_free[grid], A_free, b_free[grid], P[grid], 
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
    return Φ[1], Φ_error
end

end 
