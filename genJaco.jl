## generalized Jacobian
include("basics.jl")
include("proximal operator.jl")
include("derivation.jl")

# matrix u
function matrixU(beta::Vector{T}, kappa1::T) where T <: AbstractFloat
    U = Diagonal(zeros(length(beta), length(beta)))
    for i in 1:length(beta)
        if abs(beta[i]) > kappa1
            U[i,i] = 1
        else
            U[i,i] = 0
        end
    end
    return U
end

# matrix v
function matrixV(beta::Vector{T}, kappa2::T) where T <: AbstractFloat
    nb = norm(beta)
    if (nb > kappa2)
        return (1 - kappa2/nb) .* Matrix(I, length(beta), length(beta)) + (kappa2 / nb^3) .* beta * transpose(beta)
    else
        return zeros(length(beta), length(beta))
    end
end

# generalized Jacobian of sparse group Lasso
function sparsegroupgenJaco(u::Vector{T}, X::AbstractMatrix{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat
    betatilde = betak - transpose(X) * u ./ sigmak
    ytilde = yk + u ./ tauk

    H = zeros(length(yk), length(yk))
    for l in 1:maximum(group)
        beta_l = betatilde[@view(group[:,1]) .== l]
        X_l = X[:, @view(group[:,1]) .== l]

        H .+= X_l * matrixV(proxL1norm(beta_l, lambda * w_1/sigmak), lambda * w_2 * w_l[l] / sigmak) * matrixU(beta_l, lambda * w_1/sigmak) * transpose(X_l)
    end
    H .*= (1/sigmak)

    H .+= (1/tauk) * matrixV(ytilde, 1/tauk)

    return H
end

# Matrix Σ
function matrixSig(beta::Vector{T}, kappa2::T) where T <: AbstractFloat
    notice = multB(proxBnorm(beta, kappa2))
    Sigma = Diagonal(zeros(length(beta) - 1, length(beta) - 1))
    for i in 1:(length(beta)-1)
        if abs(notice[i]) < 1e-7
            Sigma[i,i] = 1
        else
            Sigma[i,i] = 0
        end
    end

    return Sigma
end

# multiply B transpose in front of matrix
function mapBt(Z::AbstractMatrix)
    (N, n) = size(Z)
    output = zeros((N+1), n)
    for i in 1:n
        output[:,i] .= multBt(@view(Z[:,i]))
    end
    return output
end

# multiple B back
function Bmap(Z::AbstractMatrix{T}) where T <: AbstractFloat
    (N, n) = size(Z)
    output = zeros(N, (n+1))
    for i in 1:N
        output[i, 1] = Z[i, 1]
        output[i, 2:n] .= @view(Z[i, 1:(n-1)]) - @view(Z[i, 2:n])
        output[i, n+1] = -Z[i, n] 
    end
    return output
end


# Matrix w
function matrixW(beta::Vector{T}, kappa2::T) where T <: AbstractFloat
    W = Matrix(I, length(beta), length(beta))
    pin = pinv(Bmap(matrixSig(beta, kappa2)) * mapBt(matrixSig(beta, kappa2)))
    W -= mapBt(Bmap(pin))

    return W
end

# generalized Jacobian of fused Lasso
function fusedgenJaco(u::Vector{T}, X::AbstractMatrix{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T) where T <: AbstractFloat
    betatilde = betak - transpose(X) * u ./ sigmak
    ytilde = yk + u./tauk

    H = zeros(length(yk), length(yk))
    H += X * matrixU(proxBnorm(betatilde, lambda * w_2/sigmak), lambda*w_1/sigmak) * matrixW(betatilde, lambda*w_2/sigmak) * transpose(X) ./ sigmak
    H += matrixV(ytilde, 1/tauk) / tauk

    return H
end