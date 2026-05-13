import SimpleFiniteElements.Utils: barycentric
import ElasticityQMC: extrapolate!
import Printf: @printf

const AFloat = AbstractFloat

struct Triangle{T <: AFloat}
    coord::Matrix{T}
end

const succ = [2, 3, 1]
const pred = [3, 1, 2]

function area(coord::Matrix{T}) where T <: AFloat
    b = Matrix{T}(undef, 2, 3)
    b[1,1] = coord[1,1] - coord[1,3]
    b[2,1] = coord[2,1] - coord[2,3]
    b[1,2] = coord[1,2] - coord[1,3]
    b[2,2] = coord[2,2] - coord[2,3]
    return abs(b[1,1] * b[2,2] - b[1,2] * b[2,1]) / 2
end

function subdivide(coord::Matrix{T}, levels::Int64) where T <: AFloat
    @assert size(coord) == (2, 3)
    tri = Vector{Vector{Triangle}}(undef, levels)
    tri[1] = [Triangle(coord)]
    pow4 = 1
    for level = 2:levels
        parent = tri[level-1]
        pow4 *= 4
        tri[level] = Vector{Triangle}(undef, pow4)
        r = 0
        for j in eachindex(parent)
            pc = parent[j].coord
            midpt = zeros(T, 2, 3)
            for k = 1:3
                midpt[:,k] = (pc[:,pred[k]] + pc[:,succ[k]]) / 2
            end
            for k = 1:3
                r += 1
                next_tri = zeros(T, 2, 3)
                next_tri[:,1] = pc[:,k]
                next_tri[:,2] = midpt[:,pred[k]]
                next_tri[:,3] = midpt[:,succ[k]]
                tri[level][r] = Triangle(next_tri)
            end
            r += 1
            tri[level][r] = Triangle(midpt)
        end
    end
    return tri
end


function quad(tri::Vector{Vector{Triangle}}, f::Function) 
    levels = length(tri)
    T = eltype(tri[1][1].coord)
    Q = zeros(T, levels)
    A = area(tri[1][1].coord)
    for level in eachindex(Q)
        N = length(tri[level])
        Σ = 0.0
        for r = 1:N
            for k = 1:3
                Σ += f(tri[level][r].coord[:,k])
            end
        end
        Q[level] += (A/3) * Σ
        A /= 4
    end
    return Q
end

T = BigFloat
coord = zeros(T, 2, 3)
coord[:,1] = [0, 0]
coord[:,2] = [1, 0]
coord[:,3] = [0, 1]
@printf("Initial triangle has vertices ")
for k = 1:3
    @printf("(%g, %g)", coord[1,k], coord[2,k])
    if k < 3
        @printf(", ")
    end
end
@printf(".\n")

f(x) = cos(x[1]) * exp(-x[2])          
∫f = (sin(one(T)) - cos(one(T)) + exp(-one(T))) / 2

nlevels = 8
tri = subdivide(coord, nlevels)

Q = quad(tri, f)
xtable = zeros(T, nlevels, nlevels)
xtable[:,1] = Q
Δ = extrapolate!(xtable, 2, 2)

@printf("Quadrature over subtriangles:\n\n")
@printf("%5s  %8s  %14s  %6s\n\n", "level", "triangles", "Q[level]", "rate")
for level in eachindex(Q)
    if level == 1
        @printf("%5d  %8d  %14.10f\n", 
                level, length(tri[level]), Q[level])
    else
        ratio = (Q[level-1] - ∫f) / (Q[level] - ∫f)
        rate = log2(ratio)
        @printf("%5d  %8d  %14.10f  %6.3f\n", 
                level, length(tri[level]), Q[level], rate)
    end
end

@printf("\n\nExtrapolation table (up to 5x5):\n\n")
for level = 1:5
    for col = 1:level
        @printf("%14.10f ", xtable[level,col])
    end
    @printf("\n")
end

@printf("\n\nExtrapolation table errors:\n\n")
for level = 1:nlevels
    for col = 1:level
        @printf("%8.1e ", xtable[level,col] - ∫f)
    end
    @printf("\n")
end

@printf("\n\nExtrapolation table error rates:\n\n")
for level = 2:nlevels
    for col = 1:level-1
        ratio = (xtable[level-1,col] - ∫f) / (xtable[level,col] - ∫f)
        if ratio > 0
            rate = log2(ratio)
        else
            rate = NaN
        end
        @printf("%8.3f ", rate)
    end
    @printf("\n")
end

@printf("\n\nCorrection terms:\n\n")
for row = 1:nlevels-1
    for col = 1:row
        @printf("%10.2e ", Δ[row,col])
    end
    @printf("\n")
end

@printf("\n\nCorrection term rates:\n\n")
for row = 2:nlevels-1
    for col = 1:row-1
        ratio = Δ[row-1,col] / Δ[row,col]
        if ratio > 0
            rate = log2(ratio)
        else
            rate = NaN
        end
        @printf("%8.4f ", rate)
    end
    @printf("\n")
end

