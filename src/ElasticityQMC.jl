module ElasticityQMC

import SimpleFiniteElements
import StaticArrays: SA
#using OffsetArrays
import FFTW
import SparseArrays

export IdxPair, PDEStore, InterpolationStore
export SPOD_points, pcg!, extrapolate!, extrapolate, check_rates
export double_indices, interpolated_K!, interpolated_μ!
#export slow_integrand!
#export slow_simulations!
export integrand_random_K!, integrand_random_K_μ!, simulations_random_K!,
       slow_integrand_random_K_μ!
export simulations_random_K_μ!, slow_simulations_random_K_μ!
export soln_filename, save_soln, sum_then_extrapolate, extrapolate_then_sum
 
const IdxPair = Tuple{Int64, Int64}
const Vec64 = Vector{Float64}
const Mat64 = Matrix{Float64}
const SparseCholeskyFactor = SparseArrays.CHOLMOD.Factor{Float64}
const AVec64 = AbstractVector{Float64}
const FuncOrConst = Union{Function,Float64}
const DOF = SimpleFiniteElements.DegreesOfFreedom

struct PDEStore
    conforming::Bool
    Λ::Float64
    dof::DOF
    b_free::Vec64
    u_free_det::Vec64
    solver::Symbol
    P::SparseCholeskyFactor
    tol::Float64    # used by PCG solver
    maxits::Integer
    wkspace::Mat64
end

struct InterpolationStore 
    idx::Vector{IdxPair}
    α::Float64
    M_α::Float64
    coef::Mat64   # Coefficients in the KL expansion.
    vals::Mat64   # Expansion values at the interpolation points.
    ∂₁coef::Mat64 # Coefficients and values for the partial derivatives
    ∂₁vals::Mat64 # of the expansion (only needed for μ).
    ∂₂coef::Mat64
    ∂₂vals::Mat64
    plan::FFTW.r2rFFTWPlan
end

function SPOD_points end
function pcg! end
function extrapolate! end
function extrapolate end
function check_rates end
function soln_filename end
function save_soln end
function sum_then_extrapolate end
function extrapolate_then_sum end
include("submodules/Utils.jl")

function double_indices end
function interpolated_K! end
function interpolated_μ! end
include("submodules/InterpolatedCoefs.jl")

function integrand_random_K! end
function integrand_random_K_μ! end
function slow_integrand_random_K_μ! end
include("submodules/PDE.jl")

function simulations_random_K! end
function simulations_random_K_μ! end
function slow_simulations_random_K_μ! end
include("submodules/QMC.jl")

end # module Elasticity_QMC
