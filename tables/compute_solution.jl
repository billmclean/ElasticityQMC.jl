using SimpleFiniteElements
using ElasticityQMC
import StaticArrays: SA
using Printf
using JLD2

domain_path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(domain_path)

hmax = 0.2
conforming_elements = false
solver = :pcg  # :direct or :pcg
pcg_tol = 1e-10

if conforming_elements
    mesh_order = 1
    ngrids = 4
    element_description = "conforming"
else
    mesh_order = 2
    ngrids = 4
    element_description = "non-conforming"
end
mesh = FEMesh(gmodel, hmax, order=mesh_order, save_msh_file=false, 
	      refinements=ngrids-1, verbosity=2)
h_finest = max_elt_diameter(mesh[ngrids])
h_string = @sprintf("%0.4f", h_finest)

pstore = PDEStore(mesh; conforming=conforming_elements, solver=solver,
                  pcg_tol=pcg_tol)
num_free = 2 * pstore.dof[end].num_free

α = 2.0
if exno == 2
    n = 22 
elseif exno == 3
    n = 15
    use_fft = false 
else
    error("Unknown example number exno = $exno.")
end

idx = double_indices(n)

N_std = 256 # used for K and μ
N_hi  = 512 # used for ∇μ
istore = InterpolationStore(idx, α, (N_std, N_std), (N_hi, N_hi))

if exno == 2 # μ deterministic, K random
    s₁ = 0
    s₂ = lastindex(idx)
elseif exno == 3 # both μ and K random
    s₁ = s₂ = lastindex(idx)
end
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s₁+s₂, qmc_path)

msg1 = """
Example $exno.  Solving BVP with $element_description finite elements.
Solver is $solver with tol = $pcg_tol.
Employing $(Threads.nthreads()) threads.
SPOD QMC points with s₁ = $s₁, s₂ = $s₂, α = $α.
Constructing a family of $ngrids FEM meshes by uniform refinement.
Finest FEM mesh has $num_free degrees of freedom and h = $h_string."""

println(msg1)

if exno == 4 && !use_fft
    msg2 = """
    Evaluating K and μ directly (no FFT).
    """
else
    msg2 = """
    Using $N_std x $N_std grid to interpolate K and μ.
    Using $N_hi x $N_hi grid to interpolate ∇μ if needed.
    """
end

println(msg2)

μ(x₁, x₂) = 1 + x₁ + x₂
∇μ(x₁, x₂) = SA[1.0, 1.0]
f(x₁, x₂) = SA[1-x₂^2, 2x₁-20.0]

function get_functional_values(exno::Int64, Λ::Float64; nrows=4)
    @printf("\nExample %d: Λ = %g, nrows = %d\n", exno, Λ, nrows)
    if conforming_elements
	soln_file = "results_ex$(exno)_$(Λ)_$(ngrids)_conforming.jld2"
    else
	soln_file = "results_ex$(exno)_$(Λ)_$(ngrids)_nonconf.jld2"
    end
    if isfile(soln_file)
        @printf("Loading solution data from %s.\n", soln_file)
        msg1 = load(soln_file, "msg1")
        msg2 = load(soln_file, "msg2")
        L = load(soln_file, "L")
        L_det = load(soln_file, "L_det")
        pcg_its = load(soln_file, "pcg_its")
        elapsed = load(soln_file, "elapsed")
    else
        L = Vector{Matrix{Float64}}(undef, nrows)
        L_det = Vector{Vector{Float64}}(undef, nrows)
        pcg_its = Vector{Matrix{Int64}}(undef, nrows)
        elapsed = zeros(nrows)
        for k = 1:nrows    
            start = time()
            if exno == 2
                L[k], L_det[k], pcg_its[k] = simulations!(pts[k], Λ, μ, ∇μ, f, 
                                                          pstore, istore)
	    elseif exno == 3
	        if use_fft
                    L[k], L_det[k], pcg_its[k] = simulations!(pts[k], Λ, f, 
                                                              pstore, istore)
	        else
                    L[k], L_det[k], pcg_its[k] = slow_simulations!(pts[k], α, Λ, 
                                                                   idx, f, pstore)
	        end
            end
            elapsed[k] = time() - start
            jldsave(soln_file; msg1, msg2, L, L_det, pcg_its, elapsed)
        end
    end
    return msg1, msg2, L, L_det, pcg_its, elapsed
end
    
function create_tables(exno::Int64, Λ::Float64; nrows=4)
    msg1, msg2, L, L_det, pcg_its, elapsed = get_functional_values(exno, Λ, 
                                                                   nrows)
    println(msg1)
    println(msg2)
    for k = 1:nrows
        N = size(L[k], 2)
        EL = sum(L[k], dims=2) / N


end
