using SimpleFiniteElements
using ElasticityQMC
import ElasticityQMC: integrand_init!, integrand!
import StaticArrays: SA

α = 2.0
Λ = 1.0
n = 22
idx = double_indices(n)
s₁ = s₂ = lastindex(idx)
N_std = 256
N_hi  = 512
istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))


domain_path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(domain_path)

ngrids = 4
hmax = 0.2
mesh = FEMesh(gmodel, hmax, order=2, save_msh_file=false,
              refinements=ngrids-1, verbosity=2)
pstore = PDEStore(mesh; conforming=false, solver=:pcg, pcg_tol=1e-10)

f(x, y) = SA[1-y^2, 2x-20.0]
y = rand(s₁) .- 1/2
z = rand(s₂) .- 1/2

function profile_test(y, z)
    Φ_det, Φ_det_error = integrand_init!(pstore, Λ, f)
    Φ, Φ_error = integrand!(y, z, Λ, pstore, istore)
end
