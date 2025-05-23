module ElasticityQMC

import SimpleFiniteElements
import StaticArrays: SA
#using OffsetArrays
import FFTW
import SparseArrays

export IdxPair, PDEStore, InterpolationStore
export SPOD_points, pcg!, extrapolate!, extrapolate, check_rates
export double_indices, interpolated_K!, interpolated_μ!
export integrand_init!, integrand!, slow_integrand!
export simulations!, slow_simulations!

const IdxPair = Tuple{Int64, Int64}
const Vec64 = Vector{Float64}
const Mat64 = Matrix{Float64}
const SparseCholeskyFactor = SparseArrays.CHOLMOD.Factor{Float64}
const AVec64 = AbstractVector{Float64}
const FuncOrConst = Union{Function,Float64}

struct PDEStore
    conforming::Bool
    solver::Symbol
    dof::Vector{SimpleFiniteElements.DegreesOfFreedom}
    b_free::Vector{Vec64}
    u_free_det::Vector{Vec64}
    P::Vector{SparseCholeskyFactor}
    wkspace::Vector{Mat64}
    u_free::Vector{Vec64}
    u2h::Vector{Vec64}
    pcg_tol::Float64
    pcg_maxits::Integer
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
include("submodules/Utils.jl")

function double_indices end
function interpolated_K! end
function interpolated_μ! end
include("submodules/InterpolatedCoefs.jl")

function integrand_init! end
function integrand! end
function slow_integrand! end
include("submodules/PDE.jl")

function simulations! end
function slow_simulations! end
include("submodules/QMC.jl")

end # module Elasticity_QMC
