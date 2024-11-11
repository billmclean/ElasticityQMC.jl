# Run table_1_2.jl first to define λ, μ etc.
#
using PyPlot
using Printf

ν(x, y) = λ(x, y, α, Λ) / ( 2 * (λ(x, y, α, Λ) + μ(x, y, β)) )

x = range(0, π, 201)
y = range(0, π, 201)

Z = Float64[ ν(xx, yy) for yy in y, xx in x ]

figure(1)
contourf(x, y, Z)
xlabel(L"$x_1$")
ylabel(L"$x_2$")
xticks([0, π/4, π/2, 3π/4, π], 
       [L"$0$", L"$\pi/4$", L"$\pi/2$", L"$3\pi/4$", L"\pi"])
yticks([0, π/4, π/2, 3π/4, π], 
       [L"$0$", L"$\pi/4$", L"$\pi/2$", L"$3\pi/4$", L"\pi"])
grid(true)
colorbar()

x_vals = range(0, π, 2_000)
y_vals = copy(x)
lo = Inf
hi = -Inf
for x in x_vals, y in y_vals
    global lo, hi
    next = ν(x, y)
    if next < lo
        lo = next
    end
    if next > hi
        hi = next
    end
end
@printf("Poisson ration: %0.4f ≤ ν ≤ %0.5f\n", lo, hi)
