### totally new version
using LinearAlgebra
using SparseArrays
using Dates
using IterativeSolvers
using Distributions
using ToeplitzMatrices
using Random
using DelimitedFiles
# Required Packages

# Basic Calculations
# multiply B
function multB(z::Vector)
    n = length(z)
    return @view(z[1:(n-1)]) - @view(z[2:n])
end

# multiply B transpose
function multBt(z::AbstractArray)
    n = length(z)
    Bz = zeros(n + 1)
    Bz[1] = z[1]
    Bz[n+1] = -z[n]
    Bz[2:n] = @view(z[2:n]) - @view(z[1:(n-1)])
    return Bz
end

# find BB^t with size n
function BBt(n)
    bbt = Tridiagonal(zeros(n, n))
    bbt[1,1] = 2
    bbt[1,2] = -1
    for i in 2:(n-1)
        bbt[i, i] = 2
        bbt[i, i - 1] = -1
        bbt[i, i + 1] = -1
    end
    bbt[n, n-1] = -1
    bbt[n, n] = 2
    return bbt
end

# multiply BB^t
function multBBt(z::Vector)
    nmo = length(z)
    bbtz = zeros(nmo)
    bbtz[1] = 2 * z[1] - z[2]
    bbtz[2:(nmo-1)] = - @view(z[1:(nmo-2)]) + 2 .* @view(z[2:(nmo-1)]) - @view(z[3:(nmo)])
    bbtz[nmo] = -z[nmo - 1] + 2 * z[nmo]
    return bbtz
end

# find number of nonzero elements
function n_non_zeros(x::Vector{T}) where T <: AbstractFloat
    sum = 0
    j = 0
    sorted = sort(abs.(x), rev=true)
    while sum < 0.999 * norm(x, 1)
        j += 1
        sum += sorted[j]
    end
    return j
end

# find positive part
function positive_part(x::Vector{T}) where T <: AbstractFloat
    y = copy(x)
    y[y .< 0.0] .= 0.0
    return y
end

# projection to interval
function interval_project(x::Vector{T}, lambda::T) where T <: AbstractFloat
    y = copy(x)
    y[y .< -lambda] .= -lambda
    y[y .> lambda] .= lambda
    return y
end

# soft thresholding
function soft_thresholding(x::Vector{T}, lambda::T) where T <: AbstractFloat
    return sign.(x).*positive_part(abs.(x).-lambda)
end

# rose algorithm
function rose(x::Vector{T}) where T <: AbstractFloat
    u = multB(x)
    nm1 = length(u)
    zhat = zeros(nm1)

    # step 1
    s = 0
    for j in 1:(nm1)
        s += j * u[j]
    end
    s = -s./(nm1 + 1)

    # step 2
    zhat[nm1] = u[nm1] + s
    for j in (nm1-1):-1:1
        zhat[j] = zhat[j+1] + u[j]
    end

    # step 3
    for j in 2:(nm1)
        zhat[j] += zhat[j-1]
    end
    
    return zhat
end

# sfa algorithm 
function sfa(v::Vector{T}, lambda2::T, z0::Vector{T}, maxiter) where T <: AbstractFloat
    L = 2 - 2 * cos(pi * (length(v) - 1)/ length(v))
    for i in 1:maxiter
        zprev = z0
        g = multBBt(z0) .- multB(v)
        z0 = interval_project(z0 - g/L, lambda2)
        if (norm(zprev - z0) < 1e-4)
            break
        end
    end
    return z0
end