module QMC

import ..Mat64, ..PDEStore, ..InterpolationStore, ..IdxPair,
       ..slow_integrand!,
       ..integrand_random_K!, ..integrand_random_K_μ!, 
       ..simulations_random_K!, ..simulations_random_K_μ!
import SimpleFiniteElements: FEMesh
using LinearAlgebra

# Extending dummy functions from ElasticityQMC.
#import ..simulations!, ..slow_simulations!

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

#function slow_simulations!(pts::Mat64, α::Float64, Λ::Float64, 
#	                   idx::Vector{IdxPair}, f::Function, pstore::PDEStore)
#    Φ_det = integrand_init!(pstore, Λ, f)
#    blas_threads = BLAS.get_num_threads()
#    BLAS.set_num_threads(1)
#    s, N = size(pts)
#    s₁ = s₂ = s ÷ 2
#    chunks = collect(Iterators.partition(1:N, N ÷ Threads.nthreads()))
#    ngrids = length(pstore.dof)
#    Φ = zeros(ngrids, N)
#    pcg_its = zeros(ngrids, N)
#    Threads.@threads for chunk in chunks
##    for chunk in chunks
#	pstore_local = deepcopy(pstore)
#	for l in chunk
#	    y = view(pts, 1:s₁, l)
#	    z = view(pts, s₁+1:s₁+s₂, l)
#	    Φ[:,l], pcg_its[:,l] = (
#                slow_integrand!(y, z, α, Λ, idx, pstore_local) )
#	end
#    end
#    BLAS.set_num_threads(blas_threads)
#    return Φ, Φ_det, pcg_its
#end

end # module QMC
