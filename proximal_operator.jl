## proximal operator
include("basics.jl")

# proxl1norm; page 16
function proxL1norm(x::Vector{T}, kappa::T) where T <: AbstractFloat
    return positive_part(abs.(x) .- kappa) .* sign.(x)
end

# proxl2norm; page 16
function proxL2norm(x::Vector{T}, kappa::T) where T <: AbstractFloat
    if norm(x) < 1e-7
        return zeros(length(x))
    else 
        return maximum([1- kappa/norm(x), 0.0]) .* x
    end
end

# proxBnorm; see Li (2018) page 8, Liu (2010) page 
function proxBnorm(v::Vector{T}, lambda2::T) where T <: AbstractFloat
    # finding z
    zhat = rose(multB(v))
    lambda2_max = norm(zhat, Inf)
    if lambda2 >= lambda2_max
        z = zhat
    else 
        # SFA, see algorithm 3 of Liu (2010)
        z = sfa(v, lambda2, zhat, 100)
    end

    # equation 18 of Liu (2010) or Lem1 of Li (2018)
    x = v - multBt(z)

    return x
end


# proxfused
function proxfused(x::Vector{T}, lambda1::T, lambda2::T) where T <: AbstractFloat
    return proxL1norm(proxBnorm(x, lambda2), lambda1)
end

# proxgroup; we follow the formula of our paper. please do not refer Klosa(2020)'s one.
# Especially, I want to use Chen (2022). I use lambda instead of w.
# Now, gamma * lambda = lambda1, (1-gamma) * lambda = lambda2, sqrt{p_l} = w_l 
# By the equation (2.7),
function proxgroup(x::Vector{T}, lambda1::T, lambda2::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat
    y = copy(x)
    for l in 1:maximum(group)
        x_l = x[@view(group[:,1]) .== l]
        eta = soft_thresholding(x_l, lambda1)

        if (norm(eta) > lambda2 * w_l[l])
            y[@view(group[:,1]) .== l] = eta
        else
            y[@view(group[:,1]) .== l] .= 0
        end
    end

    return y
end