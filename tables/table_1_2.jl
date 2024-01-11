using SimpleFiniteElements
import StaticArrays: SA
import SimpleFiniteElements.MeshGen: default_elt_dof, max_elt_diameter
import SimpleFiniteElements.Elasticity: ∫∫λ_div_u_div_v!, ∫∫2μ_εu_εv!, 
					∫∫f_dot_v!, elasticity_soln, 
                                        error_norms, visualise_soln
import SimpleFiniteElements.NonConformingElasticity as NCE
import Printf: @printf

nrows = 5 # must be at least 2
quadrature_level = 1
Λ = 1.0
#Λ = 1000.0
α = 0.5
β = 1.0
println("Parameters Λ = $Λ, α = $α, β = $β")
τ(x, y, α) = 1 + α * sin(2x)
λ(x, y, α, Λ) = Λ * τ(x, y, α)
μ(x, y, β) = 1 + β * (x + y)
μ_plus_λ(x, y, α, β, Λ) = μ(x, y, β) + λ(x, y, α, Λ)
∇μ = SA[β, β]

function exact_u(x, y, Λ)
    common_term = sin(x) * sin(y) / Λ
    u1 = ( cos(2x) - 1 ) * sin(2y) + common_term
    u2 = ( 1 - cos(2y) ) * sin(2x) + common_term
    return SA[u1, u2]
end

function ∇u(x, y, Λ)
    sx, cx = sincos(x)
    sy, cy = sincos(y)
    s2x, c2x = sincos(2x)
    s2y, c2y = sincos(2y)
    s2x_s2y = s2x * s2y
    cx_sy = cx * sy
    sx_cy = sx * cy
    ∂₁u₁ = -2s2x_s2y        + cx_sy / Λ
    ∂₂u₁ = 2(c2x - 1) * c2y + sx_cy / Λ
    ∂₁u₂ = 2(1 - c2y) * c2x + cx_sy / Λ
    ∂₂u₂ =  2s2x_s2y        + sx_cy / Λ
    return SA[ ∂₁u₁  ∂₁u₂
	       ∂₂u₁  ∂₂u₂ ]
end

function ϕ(x, y, β)
    sx, cx = sincos(2x)
    sy, cy = sincos(2y)
    return 2β * ( 2sx * sy - cx + cy ) + 4μ(x, y, β) * sy * (2cx - 1)
end

function f(x, y, α, β, Λ)
    ψ1 = μ(x, y, β) * ( 2sin(x) * sin(y) - cos(x+y) ) - β * sin(x+y)
    ψ2(x, y) = 2β * cos(x) * sin(y)
    ψ3 = τ(x, y, α) * cos(x+y)
    f1 =  ϕ(x, y, β) - ψ3 - 2α * cos(2x) * sin(x+y) + (ψ1 -  ψ2(x, y)) / Λ
    f2 = -ϕ(y, x, β) - ψ3 + (ψ1 - ψ2(y, x)) / Λ
    return SA[f1, f2]
end

gD = SA[0.0, 0.0]
essential_bcs = [("Top", gD), ("Bottom", gD), ("Left", gD), ("Right", gD)]

path = joinpath("..", "spatial_domains", "square.geo")
gmodel = GeometryModel(path)

bilinear_forms = Dict("Omega" => [(∫∫λ_div_u_div_v!, λ, α, Λ),
				  (∫∫2μ_εu_εv!, μ, β)])
linear_funcs = Dict("Omega" => (∫∫f_dot_v!, f, α, β, Λ))
hmax = 0.2
mesh = FEMesh(gmodel, hmax, order=1, save_msh_file=false,
              refinements=nrows-1, verbosity=2)

L2err = Vector{Float64}(undef, nrows)
H1err = similar(L2err)

@printf("Conforming elements.\n")
@printf("%6s& %6s& %8s& %5s& %8s& %5s\\\\\n\n",
	"h", "DoF", "L2", "rate", "H1", "rate")
for row in eachindex(L2err)
    start = time()
    dof = DegreesOfFreedom(mesh[row], essential_bcs)
    u1h, u2h = elasticity_soln(dof, bilinear_forms, linear_funcs)
    if row == 2
        scale = 15.0
        visualise_soln(dof, u1h, u2h, scale, 1)
    end
    L2err[row], H1err[row] = error_norms(u1h, u2h, exact_u, ∇u, 
					 dof, quadrature_level, Λ)
    h = max_elt_diameter(mesh[row])
    num_dof = 2 * dof.num_free
    elapsed = time() - start
    if row == 1
	@printf("%6.3f& %6d& %8.2e& %5s& %8.2e& %5s\\\\\n",
		h, num_dof, L2err[row], "",    H1err[row], "")
    else
	L2rate = log2(L2err[row-1] / L2err[row])
	H1rate = log2(H1err[row-1] / H1err[row])
	@printf("%6.3f& %6d& %8.2e& %5.3f& %8.2e& %5.3f\\\\\n",
		h, num_dof, L2err[row], L2rate,    H1err[row], H1rate)
    end
end

nc_bilinear_forms = Dict{String,Any}(
                         "Omega" => [(NCE.∫∫a_div_u_div_v!, μ_plus_λ, α, β, Λ),
                                     (NCE.∫∫μ_∇u_colon_∇v!, μ, β),
				     (NCE.correction!, ∇μ)])
nc_linear_funcs = Dict("Omega" => (NCE.∫∫f_dot_v!, f, α, β, Λ))
nc_mesh = FEMesh(gmodel, hmax, order=2, save_msh_file=false,
                  refinements=nrows-1, verbosity=2)

nc_L2err = similar(L2err)
nc_H1err = similar(L2err)

@printf("\nNon-conforming elements.\n")
@printf("%6s& %6s& %8s& %5s& %8s& %5s\\\\\n\n",
	"h", "DoF", "L2", "rate", "H1", "rate")
for row in eachindex(nc_L2err)
    start = time()
    nc_dof = DegreesOfFreedom(nc_mesh[row], essential_bcs, NCE.ELT_DOF)
    nc_u1h, nc_u2h = elasticity_soln(nc_dof, nc_bilinear_forms, nc_linear_funcs)
    nc_L2err[row], nc_H1err[row] = NCE.error_norms(nc_u1h, nc_u2h, exact_u, ∇u, 
				  	           nc_dof, quadrature_level, Λ)
    h = max_elt_diameter(mesh[row])
    nc_num_dof = 2 * nc_dof.num_free
    elapsed = time() - start
    if row == 1
	@printf("%6.3f& %6d& %8.2e& %5s& %8.2e& %5s\\\\\n",
		h, nc_num_dof, nc_L2err[row], "", nc_H1err[row], "")
    else
	nc_L2rate = log2(nc_L2err[row-1] / nc_L2err[row])
	nc_H1rate = log2(nc_H1err[row-1] / nc_H1err[row])
	@printf("%6.3f& %6d& %8.2e& %5.3f& %8.2e& %5.3f\\\\\n", h, 
		nc_num_dof, nc_L2err[row], nc_L2rate, nc_H1err[row], nc_H1rate)
    end
end

