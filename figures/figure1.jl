using ElasticityQMC
import ElasticityQMC.InterpolatedCoefs: interpolated_λ!
import ElasticityQMC.Utils: SPOD_points
using JLD2
using PyPlot
using Printf

qmc = false
N₁ = 256
N₂ = 256
n = 22
α = 2.0
Λ = 1.0
idx = double_indices(n)
s = lastindex(idx)
qmc_path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(s, qmc_path)
k = 4
@printf("Using coefficient array with n = %d, α = %0.4f, Λ = %0.4f.\n", 
        n, α, Λ)
@printf("%d points in %d dimensions\n", Nvals[k], s)
@printf("Using %d x %d spatial grid\n", N₁, N₂)
istore = InterpolationStore(idx, α, N₁, N₂)
#col = rand(collect(1:Nvals[k]))
col = 88
@printf("Choosing z from column %d\n", col)
z = pts[k][:,col]
λ = interpolated_λ!(z, istore, Λ)

#figure(1, figsize=(6,4))
figure(1)
xx = range(0, 1, length=100)
yy = range(0, 1, length=100)

contourf(xx, yy, λ.(xx',yy), 10)
axis("equal")
xlabel(L"$x_1$")
ylabel(L"$x_2$")
title(L"$\lambda(\mathbf{x},\mathbf{z})$")
colorbar()

savefig("fig1.pdf")
