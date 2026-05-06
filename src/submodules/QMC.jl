module QMC

import ..PDEStore, ..InterpolationStore, ..IdxPair
import ..integrand_random_K!, ..integrand_random_K_μ!
import ..simulations_random_K!, ..simulations_random_K_μ!
import ..Mat64
import SimpleFiniteElements: FEMesh
import LinearAlgebra: BLAS

"""
    simulations_random_K!(pts, μ, ∇μ, pstore, istore)

Computes the functional for a given set of QMC points and FEM mesh, when
the bulk modulus `K` is random (but the shear modulus `μ` is not).  
Evaluation of `K` is sped up by use of FFT and interpolation.
"""
function simulations_random_K!(pts::Mat64, μ::Function, ∇μ::Function, 
                               pstore::PDEStore, istore::InterpolationStore)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    L = zeros(N)
    pcg_its = zeros(Int64, N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    z = view(pts, :, l)
	    L[l], pcg_its[l] = integrand_random_K!(z, μ, ∇μ, 
                                                   pstore_local, istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return L, pcg_its
end

"""
    simulations_random_K!(pts, α, μ, ∇μ, idx, pstore)

This version evaluates the trigonometric series for the bulk modulus `K` 
directly (that is, not using FFT and interpolation).
"""
function simulations_random_K!(pts::Mat64, α::Float64, 
                               μ::Function, ∇μ::Function, 
                               idx::Vector{IdxPair}, pstore::PDEStore)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    L = zeros(N)
    pcg_its = zeros(Int64, N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	for l in chunk
	    z = view(pts, :, l)
	    L[l], pcg_its[l] = integrand_random_K!(z, α, μ, ∇μ, idx,
                                                   pstore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return L, pcg_its
end

"""
    simulations_random_K_μ!(pts, pstore, istore)

Computes the functional for a given set of QMC points and FEM mesh, when the
bulk modulus `K` and the shear modulus `μ` are both random.  Evaluation of 
`K` and `μ` are sped up by use of FFT and interpolation.
"""
function simulations_random_K_μ!(pts::Mat64, 
                                 pstore::PDEStore, istore::InterpolationStore)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    s₁ = s₂ = s ÷ 2
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    L = zeros(N)
    pcg_its = zeros(Int64, N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    y = view(pts, 1:s₁, l)
	    z = view(pts, s₁+1:s₁+s₂, l)
	    L[l], pcg_its[l] = integrand_random_K_μ!(y, z, 
                                                     pstore_local, istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return L, pcg_its
end

"""
    simulations_random_K_μ!(pts, α, idx, pstore)

This version evaluates the trigonometric series for the bulk modulus `K` 
and the shear modulus `μ` directly (that is, not using FFT and interpolation).
"""
function simulations_random_K_μ!(pts::Mat64, α::Float64, 
                                 idx::Vector{IdxPair}, pstore::PDEStore)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    s₁ = s₂ = s ÷ 2
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    L = zeros(N)
    pcg_its = zeros(Int64, N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	for l in chunk
	    y = view(pts, 1:s₁, l)
	    z = view(pts, s₁+1:s₁+s₂, l)
	    L[l], pcg_its[l] = integrand_random_K_μ!(y, z, α, idx,
                                                     pstore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return L, pcg_its
end

end # module QMC
