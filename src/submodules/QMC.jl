module QMC

import ..Mat64, ..PDEStore, ..InterpolationStore, ..IdxPair,
       ..integrand_init!, ..integrand!, ..slow_integrand!
using LinearAlgebra

# Extending dummy functions from ElasticityQMC.
import ..simulations!, ..slow_simulations!

function simulations!(pts::Mat64, Λ::Float64, μ::Function, ∇μ::Function,
	              f::Function, pstore::PDEStore, istore::InterpolationStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, Λ, μ, ∇μ, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    pcg_its = zeros(length(pstore.dof), N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    z = view(pts, :, l)
	    Φ[l], Φ_error[l], pcg_its[:,l] = integrand!(z, Λ, μ, ∇μ, 
							pstore_local, 
					                istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error, pcg_its
end

function simulations!(pts::Mat64, K::Function, f::Function,
	              pstore::PDEStore, istore::InterpolationStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, K, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    pcg_its = zeros(length(pstore.dof), N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    y = view(pts, :, l)
	    Φ[l], Φ_error[l], pcg_its[:,l] = integrand!(y, K, pstore_local, 
						        istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error, pcg_its
end

function simulations!(pts::Mat64, Λ::Float64, f::Function,
	              pstore::PDEStore, istore::InterpolationStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, Λ, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    s₁ = s₂ = s ÷ 2
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    pcg_its = zeros(length(pstore.dof), N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    y = view(pts, 1:s₁, l)
	    z = view(pts, s₁+1:s₁+s₂, l)
	    Φ[l], Φ_error[l], pcg_its[:,l] = integrand!(y, z, Λ, pstore_local, 
					                istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error, pcg_its
end

function slow_simulations!(pts::Mat64, α::Float64, Λ::Float64, 
	                   idx::Vector{IdxPair}, f::Function, pstore::PDEStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, Λ, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    s₁ = s₂ = s ÷ 2
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    pcg_its = zeros(length(pstore.dof), N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	for l in chunk
	    y = view(pts, 1:s₁, l)
	    z = view(pts, s₁+1:s₁+s₂, l)
	    Φ[l], Φ_error[l], pcg_its[:,l] = (
                slow_integrand!(y, z, α, Λ, idx, pstore_local) )
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error, pcg_its
end

end # module QMC
