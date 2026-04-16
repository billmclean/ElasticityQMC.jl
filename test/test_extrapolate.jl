import ElasticityQMC.Utils: extrapolate!, check_rates
import Printf: @printf

@printf("Using the trapezoidal rule to approximate ∫exp(-x)dx over [0,1].\n")
m = 6
n = 5
xtable = zeros(m, n)
f(x) = exp(-x)
∫f = 1.0 - exp(-1.0)
N = 1
for i = 1:m
    local x, s
    global N
    x = range(0, 1, N+1)
    s = f(x[1])/2
    for k = 2:N
        s += f(x[k])
    end
    s += f(x[N+1])/2
    xtable[i,1] = s / N
    N *= 2
end

correction = extrapolate!(xtable, 2)
@printf("\nExtrapolation table:\n")
for i = 1:m
    for j = 1:min(i, n)
        @printf("%14.12f ", xtable[i,j])
    end
    @printf("\n")
end
@printf("\nCorrection terms:\n")
for i = 2:m
    @printf("%14s ", "")
    for j = 2:min(i, n)
        @printf("%14.4e ", correction[i,j])
    end
    @printf("\n")
end

rate = check_rates(xtable)
@printf("\nApparent convergent rates:\n")
for i = 2:m-1
    for j = 1:min(i-1,n-1)
        @printf("%14.4f ", rate[i,j])
    end
    @printf("\n")
end
