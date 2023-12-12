## Semismooth Newton
include("basics.jl")
include("proximal operator.jl")
include("derivation.jl")
include("genJaco.jl")

## find step size for sparse group lasso
function sparsegroupstep(u::Vector{T}, f::Vector{T}, d::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat
    # hyperparameters
    rho = 0.5
    mu = 0.2

    # Initalize
    alphaj = 1
    uPsi = sparsegroupPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)
    dotd = dot(f, d)

    while (sparsegroupPsi(u + alphaj .* d, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l) > uPsi + mu * alphaj * dotd)
        alphaj *= rho
    end

    return alphaj

end

## for sparse group lasso
function sparsegroupSSN(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat
    # Requirements
    @assert sigmak > 0
    @assert tauk > 0
    
    (N, n) = size(X)
    @assert n == length(betak)
    @assert N == length(yk)

    # hyperparameters
    eta = 0.5
    varrho = 0.5
    tol = 1e-7

    # Initialize
    u = zeros(N)
    f = sparsegroupgradPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)
    j = 0

    while norm(f) > tol
        println("j: ", j)
        # step 1
        H = sparsegroupgenJaco(u, X, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)
        d = H \ (-f)

        if (norm(H*d + f) > minimum([eta, norm(f)^(1+varrho)]))
            println("you failed")
        end

        # step 2
        alphaj = sparsegroupstep(u, f, d, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)

        # step 3
        u += alphaj .* d
        f = sparsegroupgradPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)

        # step 4
        j += 1
    end

    betabar = proxgroup(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak, group, w_l)
    ybar = proxL2norm(yk + u ./ tauk, 1/tauk)

    return (betabar, ybar)

end 

## find step size for fused Lasso
function fusedstep(u::Vector{T}, f::Vector{T}, d::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T) where T <: AbstractFloat
    # hyperparameters
    rho = 0.5
    mu = 0.2

    # Initalize
    alphaj = 1
    uPsi = fusedPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk)
    dotd = dot(f, d)

    while (fusedPsi(u + alphaj .* d, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk) > uPsi + mu * alphaj * dotd)
        alphaj *= rho
    end

    return alphaj

end

## for fused lasso
function fusedSSN(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T) where T <: AbstractFloat
    # Requirements
    @assert sigmak > 0
    @assert tauk > 0
    
    (N, n) = size(X)
    @assert n == length(betak)
    @assert N == length(yk)

    # hyperparameters
    eta = 0.5
    varrho = 0.5
    tol = 1e-7

    # Initialize
    u = zeros(N)
    f = fusedgradPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk)
    j = 0

    while norm(f) > tol
        # step 1
        println("j: ", j)
        H = fusedgenJaco(u, X, lambda, w_1, w_2, betak, yk, sigmak, tauk)
        d = H \ (-f)

        if (norm(H*d + f) > minimum([eta, norm(f)^(1+varrho)]))
            println("you failed")
        end

        # step 2
        alphaj = fusedstep(u, f, d, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk)

        # step 3
        u += alphaj .* d
        f = fusedgradPsi(u, X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk)

        # step 4
        j += 1
    end

    betabar = proxfused(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak)
    ybar = proxL2norm(yk + u ./ tauk, 1/tauk)

    return (betabar, ybar)

end 