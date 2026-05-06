module PDE

using SimpleFiniteElements
import SimpleFiniteElements.FEM: average_field
import LinearAlgebra: BLAS, cholesky

import ..IdxPair, ..PDEStore, ..InterpolationStore
import ..pcg!, ..extrapolate!, ..interpolated_K!, ..interpolated_μ!
import ..InterpolatedCoefs: slow_K, slow_μ, slow_∂₁μ, slow_∂₂μ
import ..integrand_random_K!, ..integrand_random_K_μ!
import ..Vec64, ..Mat64, ..AVec64, ..SA, ..DOF, ..SparseCholeskyFactor

"""
    PDEStore(mesh, conforming, μ₀, ∇μ₀, f, solver, pcg_tol, pcg_maxits)

Constructor for a PDEStore.      
"""
function PDEStore(mesh::FEMesh, conforming::Bool, Λ::Float64, 
                  μ₀::Function, ∇μ₀::Function, f::Function, 
                  solver::Symbol, pcg_tol::Float64, pcg_maxits::Int64)
    gD = SA[0.0, 0.0]
    essential_bcs = [("Top", gD), ("Bottom", gD), ("Left", gD), ("Right", gD)]
    if conforming
        El = SimpleFiniteElements.Elasticity
        dof = DegreesOfFreedom(mesh, essential_bcs)
        bilinear_forms = Dict("Omega" => [(El.∫∫λ_div_u_div_v!, Λ),
                              (El.∫∫2μ_εu_εv!, μ₀)])
    else
        El = SimpleFiniteElements.NonConformingElasticity
        dof = DegreesOfFreedom(mesh, essential_bcs, El.ELT_DOF)
        bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, Λ),
                                          (El.∫∫μ_∇u_colon_∇v!, μ₀),
                                          (El.correction!, ∇μ₀)])
    end
    linear_funcs = Dict("Omega" => (El.∫∫f_dot_v!, f))
    A_free, _ = assemble_matrix(dof, bilinear_forms, 2)
    b_free, _ = assemble_vector(dof, linear_funcs, 2)
    P = cholesky(A_free)    # Use as a preconditioner
    u_free_det = P \ b_free # Deterministic solution (y = z = 0)
    num_free = dof.num_free
    wkspace = Matrix{Float64}(undef, 2num_free, 4)
    return PDEStore(conforming, Λ, dof, b_free, u_free_det, solver, P,
                    pcg_tol, pcg_maxits, wkspace)
end

"""
    random_solve!(bilinear_forms, pstore)

Solve one realisation of the PDE with random coefficients.
"""
function random_solve!(bilinear_forms::Dict, pstore::PDEStore)
    (; dof, solver, b_free, u_free_det, P, tol, maxits, wkspace) = pstore
    A_free, _ = assemble_matrix(dof, bilinear_forms, 2)
    if solver == :direct
        u_free = A_free \ b_free
    elseif solver == :pcg
        u_free = copy(u_free_det)
        num_its = pcg!(u_free, A_free, b_free, P, tol, maxits, wkspace)
    else
        error("Unknown solver")
    end
    return u_free, num_its
end

"""
    L_functional(u_free, dof)

Evaluate the linear functional given the vector `u_free` of finite element
solution values at the free nodes.
"""
function L_functional(u_free::Vector{Float64}, dof::DOF)
    num_free, num_fixed = dof.num_free, dof.num_fixed
    u2h = zeros(num_free + num_fixed) # 2nd component of uh
    for k = 1:num_free
        u2h[k] = u_free[num_free+k]
    end
    L = average_field(u2h, "Omega", dof)
    return L
end

"""
    integrand_random_K!(z, μ, ∇μ, pstore, istore)

Evaluates the linear functional for a given QMC point `z` when the bulk
modulus `K` is random (but the shear modulus `μ` is not).  This version
uses FFT and interpolation to speed up evaluation of `K`.
"""
function integrand_random_K!(z::AVec64, μ::Function, ∇μ::Function,
                            pstore::PDEStore, istore::InterpolationStore)
    (; conforming, Λ, dof, b_free, tol, maxits) = pstore
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
    u_free, num_its = random_solve!(bilinear_forms, pstore)
    L, _ = L_functional(u_free, dof)
    return L, num_its
end

"""
    integrand_random_K!(y, z, α, μ, ∇μ, idx, pstore)

Computes `K` directly, without using FFT and interpolation.
"""
function integrand_random_K!(z::AVec64, α::Float64, 
                                  μ::Function, ∇μ::Function, 
                                  idx::Vector{IdxPair}, pstore::PDEStore)
    (; conforming, Λ, dof, b_free, tol, maxits) = pstore
    K(x₁, x₂) = slow_K(x₁, x₂, z, α, Λ, idx)
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
    u_free, num_its = random_solve!(bilinear_forms, pstore)
    L, _ = L_functional(u_free, dof)
    return L, num_its
end

"""
    integrand_random_K_μ!(y, z, pstore, istore)

Evaluates the linear functional for a given QMC point `(y, z)` when the 
both the bulk modulus `K` and the shear modulus `μ` are random.  This version
uses FFT and interpolation to speed up evaluation of `K`, `μ` and `∇μ`.
"""
function integrand_random_K_μ!(y::AVec64, z::AVec64, 
                               pstore::PDEStore, istore::InterpolationStore)
    (; conforming, Λ, dof) = pstore
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
    u_free, num_its = random_solve!(bilinear_forms, pstore)
    L, _ = L_functional(u_free, dof)
    return L, num_its
end

"""
    integrand_random_K_μ!(y, z, α, idx, pstore)

Computes `K` and `μ` directly, without using FFT and interpolation.
"""
function integrand_random_K_μ!(y::AVec64, z::AVec64, α::Float64, 
                                    idx::Vector{IdxPair}, pstore::PDEStore)
    (; conforming, Λ, dof) = pstore
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
    u_free, num_its = random_solve!(bilinear_forms, pstore)
    L, _ = L_functional(u_free, dof)
    return L, num_its
end

end 
