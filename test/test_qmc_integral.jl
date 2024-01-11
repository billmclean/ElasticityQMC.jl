using Elasticity_QMC
using Printf

path = joinpath("..", "qmc_points", "SPOD_dim256.jld2")
Nvals, pts = SPOD_points(256, path)
for k in eachindex(Nvals)
    pts[k] .+= 1/2  # Restore to [0,1]^s
end
s1 = 3

f(x) = exp(x[1] - x[2] + x[3])
If = (ℯ - 1)^3 / ℯ

If_QMC = zeros(10)
If_MC = zeros(10)
@printf("Example in %d dimensions\n\n", s1)
@printf("%6s  %12s  %10s  %10s  %10s\n\n", "N", "QMC", "Error", "MC", "Error")
for k in eachindex(Nvals)
    N = Nvals[k]
    z_QMC = pts[k][1:s1,:]
    z_MC = rand(3, N)
    Σ_QMC = 0.0
    Σ_MC = 0.0
    for j = 1:N
	Σ_QMC += f(z_QMC[:,j])
	Σ_MC += f(z_MC[:,j])
    end
    If_QMC[k] = Σ_QMC / N
    If_MC[k] = Σ_MC / N
    @printf("%6d  %12.8f  %10.2e  %10.5f  %10.2e\n", 
	    N, If_QMC[k], If_QMC[k] - If, If_MC[k], If_MC[k] - If)
end
@printf("\n%6s  %12.8f  %10s  %10.5f\n", "Exact", If, "", If)

s2 = 256
function g(x)
    s = lastindex(x)
    Σ = 0.0
    for j = 1:s-1
	Σ += 1 / ( j^2 * (1 + x[j]) * (1 + x[j+1]) )
    end
    Σ += 1 / ( s^2 * (1 + x[s]) * (1 + x[1]) )
    return Σ
end

Ig = 0.0
for j = 1:s2
    global Ig
    Ig += 1 / j^2
end
Ig *= log(2)^2

Ig_QMC = zeros(12)
Ig_MC = zeros(12)
@printf("\n\nExample in %d dimensions\n\n", s2)
@printf("%6s  %12s  %10s  %10s  %10s\n\n", 
	"N", "QMC", "Error", "MC", "Error")
for k in eachindex(pts)
    N = Nvals[k]
    z_QMC = pts[k] 
    z_MC = rand(s2, N)
    Σ_QMC = 0.0
    Σ_MC = 0.0
    for j = 1:N
	Σ_QMC += g(z_QMC[:,j])
	Σ_MC += g(z_MC[:,j])
    end
    Ig_QMC[k] = Σ_QMC / N
    Ig_MC[k] = Σ_MC / N
    @printf("%6d  %12.8f  %10.2e  %10.5f  %10.2e\n", 
	    N, Ig_QMC[k], Ig_QMC[k] - Ig, Ig_MC[k], Ig_MC[k] - Ig)
end
@printf("\n%6s  %12.8f  %10s  %10.5f\n", "Exact", Ig, "", Ig)

