import FFTW: plan_r2r, RODFT00
import Interpolations: linear_interpolation
using PyPlot
using Printf

n = 15
idx = double_indices(n)
s₁ = lastindex(idx)
@printf("Using n = %d so s₁ = %d\n", n, s₁)
α = 2
z = rand(s₁) .- 1/2
N₁ = 255
N₂ = 127
@printf("Grid is %d x %d\n", N₁+1, N₂+1)
coef = zeros(N₁, N₂)
for j in eachindex(idx)
    k, l = idx[j]
    coef[k,l] = z[j] / (k + l)^(2α)
end
plan = plan_r2r(coef, RODFT00)
start = time()
K_fft = plan * coef
elapsed_fft = time() - start
@printf("FFT method: %0.4e secs\n", elapsed_fft)
K_slow = similar(K_fft)
start = time()
for i1 = 1:N₁
    for i2 = 1:N₂
	Σ = 0.0
	for j in eachindex(idx)
            k, l = idx[j]
	    decay_factor = 1 / (k + l)^(2α)
	    Σ += z[j] * decay_factor * sinpi(k*i1/(n+1)) * sinpi(l*i2/(n+1))
	end
        K_slow[i1,i2] = 4Σ
    end
end
elapsed_slow = time() - start
@printf("Slow method: %0.4e secs\n", elapsed_slow)

x = range(0, 1, length=N₁+2)
y = range(0, 1, length=N₂+2)
A = zeros(N₁+2, N₂+2)
A[2:N₁+1,2:N₂+1] .= K_slow
K_interp = linear_interpolation((x, y), A)

figure(1)
contourf(K_fft)
colorbar()
