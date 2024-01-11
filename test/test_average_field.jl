using FinElt
import FinElt.FEM: average_field
import StaticArrays: SA

path = joinpath("..", "spatial_domains", "unit_square.geo")
gmodel = GeometryModel(path)
conforming_elements = false
if conforming_elements
    mesh_order = 1
    El = FinElt.Elasticity
else
    mesh_order = 2
    El = FinElt.NonConformingElasticity
end
hmax = 0.2
mesh = FEMesh(gmodel, hmax, order=mesh_order, save_msh_file=false,
              verbosity=2)
gD = SA[0.0, 0.0]
essential_bcs = [("Top", gD), ("Bottom", gD), ("Left", gD), ("Right", gD)]
dof = DegreesOfFreedom(mesh, essential_bcs, El.ELT_DOF)

uh = rand(dof.num_free + dof.num_fixed)
avg, domain_area = average_field(uh, "Omega", dof)
