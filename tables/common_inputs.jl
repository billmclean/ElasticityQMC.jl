using SimpleFiniteElements
using Elasticity_QMC
import Elasticity_QMC.Utils: SPOD_points
import StaticArrays: SA
using Printf
using JLD2

domain_path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(domain_path)

hmax = 0.2
conforming_elements = false
solver = :pcg  # :direct or :pcg

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
                  pcg_tol=1e-10)
num_free = 2 * pstore.dof[end].num_free

α = 2.0
n = 22 
idx = double_indices(n)
N₁ = 255
N₂ = 255

istore = InterpolationStore(idx, α, N₁, N₂)

s = lastindex(idx)
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s, qmc_path)

msg = """
Example $exno.  Solving BVP with $element_description finite elements.
Solver is $solver.
Employing $(Threads.nthreads()) threads.
Using s = $s, α = $α.
Interpolating random coefficient using $N₁ x $N₂ spatial grid.
Constructing a family of $ngrids FEM meshes by uniform refinement.
Finest FEM mesh has $num_free degrees of freedom and h = $h_string."""

println(msg)

λ̂(x₁, x₂) = 1 + 0.5 * sinpi(2x₁) 
μ(x₁, x₂) = 1 + x₁ + x₂
∇μ(x₁, x₂) = SA[1.0, 1.0]

function create_tables(exno::Int64; Λ::Float64, nrows=4)
    @printf("\nExample %d: Λ = %g\n", exno, Λ)
    ref_row = nrows + 2
    Nref = Nvals[ref_row]
    f(x₁, x₂) = SA[1-x₂^2, 2x₁-20.0]
    if conforming_elements
	refsoln_file = "reference_soln_ex$(exno)_$(Λ)_conforming.jld2"
    else
	refsoln_file = "reference_soln_ex$(exno)_$(Λ)_nonconforming.jld2"
    end
    if isfile(refsoln_file)
        @printf("Loading reference solution (N = %d) from %s.\n", 
                Nref, refsoln_file)
        L_ref = load(refsoln_file, "L_ref")
        L_det = load(refsoln_file, "L_det")
        FEM_error = load(refsoln_file, "FEM_error")
    else
        @printf("Computing reference solution (N = %d) ...", Nref)
        start = time()
        if exno == 2
            z = pts[ref_row]
            Φ, Φ_error, Φ_det, _ = simulations!(z, Λ, μ, ∇μ, f,
                                                pstore, istore)
        elseif exno == 3
            y = pts[ref_row]
	    λ(x₁, x₂) = Λ * λ̂(x₁, x₂)
            Φ, Φ_error, Φ_det, _ = simulations!(y, λ, f, pstore, istore)
        end
        elapsed_ref = time() - start
        @printf(" in %d seconds.\n", elapsed_ref)
        L_ref = sum(Φ) / Nref
        L_det = sum(Φ_det) / Nref
        FEM_error = sum(Φ_error) / Nref
        jldsave(refsoln_file; L_ref, L_det, FEM_error)
    end
    @printf("FEM error of order %0.2e\n", FEM_error)
    @printf("\n%6s  %14s  %10s  %8s  %8s\n\n",
            "N", "L", "error", "rate", "seconds")
    L = zeros(nrows)
    L_error = similar(L)
    elapsed = similar(L)
    for k = 1:nrows    
        start = time()
        if exno == 2
            z = pts[k]
            Φ, _, _, _ = simulations!(z, Λ, μ, ∇μ, f, pstore, istore)
        elseif exno == 3
            y = pts[k]
            Φ, _, _, _ = simulations!(y, λ, f, pstore, istore)
        end
        elapsed[k] = time() - start
        L[k] = sum(Φ) / Nvals[k]
        L_error[k] = L[k] - L_ref
        if k == 1
             @printf("%6d  %14.10f  %10.3e  %8s  %8.3f\n",
                     Nvals[k], L[k], L_error[k], "", elapsed[k])
        else
            rate = log2(L_error[k-1]/L_error[k])
            @printf("%6d  %14.10f  %10.3e  %8.3f  %8.3f\n",
                    Nvals[k], L[k], L_error[k], rate, elapsed[k])
        end
    end
    @printf("\n%6d  %14.10f  %s\n", Nref, L_ref, "Reference value")
    @printf("\nLaTeX-ready version:\n%6s  %14s  %10s  %8s\n\n",
            "N", "L", "error", "rate")
    for k = 1:nrows
        if k == 1
             @printf("%6d& %14.10f& %10.2e& %8s\\\\\n",
                     Nvals[k], L[k], L_error[k], "")
        else
            rate = log2(L_error[k-1]/L_error[k])
            @printf("%6d& %14.10f& %10.2e& %8.3f\\\\\n",
                    Nvals[k], L[k], L_error[k], rate)
        end
    end
end




        
    
