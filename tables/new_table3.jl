using SimpleFiniteElements
using ElasticityQMC
import StaticArrays: SA
using Printf
using JLD2

domain_path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(domain_path)

hmax = 0.2
conforming_elements = false
solver = :direct  # :direct or :pcg
pcg_tol = 1e-10

mesh_order = 2
ngrids = 5
element_description = "non-conforming"

mesh = FEMesh(gmodel, hmax, order=mesh_order, save_msh_file=false,
              refinements=ngrids-1, verbosity=2)
h_finest = max_elt_diameter(mesh[ngrids])
h_string = @sprintf("%0.4f", h_finest)

pstore = PDEStore(mesh; conforming=false, solver=solver,
                  pcg_tol=pcg_tol)
num_free = 2 * pstore.dof[end].num_free

Λ = 1_000

n = 15
idx = double_indices(n)
α = 2.0

N_std = 256 # used for λ and μ
N_hi  = 512 # used for ∇μ
istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))

s₁ = s₂ = lastindex(idx)
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s₁+s₂, qmc_path)

λ_ = interpolated_λ!(z, istore, Λ)
λ = (x₁, x₂) -> λ_(x₁, x₂)
μ_, μ_plus_λ_, ∂₁μ, ∂₂μ = interpolated_μ!(y, istore, λ)
μ = (x₁, x₂) -> μ_(x₁, x₂)

El = SimpleFiniteElements.NonConformingElasticity
El = SimpleFiniteElements.NonConformingElasticity
μ_plus_λ = (x₁, x₂) -> μ_plus_λ_(x₁, x₂)
∇μ(x₁, x₂) = SA[ ∂₁μ(x₁, x₂), ∂₂μ(x₁, x₂) ]
bilinear_forms = Dict("Omega" => [(El.∫∫a_div_u_div_v!, μ_plus_λ),
                                  (El.∫∫μ_∇u_colon_∇v!, μ),
                                  (El.correction!, ∇μ)])

Φ = zeros(ngrids, ngrids)
for grid in eachindex(dof)
    A_free, + = assemble_matrix(dof[grid], bilinear_forms, 2)
    u_free[grid] .= A_free \ b_free[grid]
    num_free = dof[grid].num_free
    for k = 1:num_free
        u2h[grid][k] = u_free[grid][num_free+k]
    end
    Φ[grid,1], _ = average_field(u2h[grid], "Omega", dof[grid])
end

