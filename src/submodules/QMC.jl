module QMC

import ..Mat64, ..PDEStore, ..InterpolationStore
import ..PDE: integrand_init!, integrand!
using LinearAlgebra

function simulations!(pts::Mat64, Λ::Float64, μ::Function, ∇μ::Function,
	              f::Function, pstore::PDEStore, istore::InterpolationStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, Λ, μ, ∇μ, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    z = view(pts, :, l)
	    Φ[l], Φ_error[l] = integrand!(z, Λ, μ, ∇μ, pstore_local, 
					  istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error
end

function simulations!(pts::Mat64, λ::Function, f::Function,
	              pstore::PDEStore, istore::InterpolationStore)
    Φ_det, Φ_det_error = integrand_init!(pstore, λ, f)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    s, N = size(pts)
    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
    Φ = zeros(N)
    Φ_error = zeros(N)
    Threads.@threads for chunk in chunks
#    for chunk in chunks
	pstore_local = deepcopy(pstore)
	istore_local = deepcopy(istore)
	for l in chunk
	    y = view(pts, :, l)
	    Φ[l], Φ_error[l] = integrand!(y, λ, pstore_local, istore_local)
	end
    end
    BLAS.set_num_threads(blas_threads)
    return Φ, Φ_error, Φ_det, Φ_det_error
end

end # module QMC