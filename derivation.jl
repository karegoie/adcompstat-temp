## Functions from proximal operators
include("basics.jl")
include("proximal operator.jl")

# moreau envelope of sparse group Lasso
function sparsegroupMoreau(x::Vector{T}, lambda1::T, lambda2::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat
    argmin = proxgroup(x, lambda1, lambda2, group, w_l)
    pbetak = 0
    for l in 1:maximum(group)
        pbetak += w_l[l] * norm(argmin[@view(group[:,1]) .== l], 1)
    end
    pbetak *= lambda2

    pbetak += lambda1 * norm(argmin, 1)
    pbetak += 0.5 * (norm(argmin - x, 2)^2)

    return pbetak
end

# moreau envelope of fused Lasso
function fusedMoreau(x::Vector{T}, lambda1::T, lambda2::T) where T <: AbstractFloat
    argmin = proxfused(x, lambda1, lambda2)
    pbetak = 0
    pbetak += 0.5 * (norm(argmin - x, 2)^2)
    pbetak += lambda1 * norm(argmin, 1)
    pbetak += lambda2 * norm(multB(argmin), 1)

    return pbetak
end

# moreau envelope of L2 norm
function L2moreau(x::Vector{T}, kappa::T) where T <: AbstractFloat
    argmin = proxL2norm(x, kappa)
    return kappa * norm(argmin) + 0.5 * (norm(argmin - x, 2)^2)
end

# sparse group lasso Ψ
function sparsegroupPsi(u::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T, group::Vector{Int64}, w_l) where T <: AbstractFloat
    # lambda1 = lambda * w_1 / sigmak
    # lambda2 = lambda * w_2 / sigmak
    psimple = norm(u)^2 / (2 * tauk) + norm(transpose(X) * u)^2 / (2 * sigmak) - dot(u, X * betak - yk - Y)
    psimple -= sigmak * sparsegroupMoreau(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak, group, w_l)
    psimple -= tauk * L2moreau(yk + u./tauk, 1/tauk)

    return psimple
end

# fused lasso Ψ
function fusedPsi(u::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T) where T <: AbstractFloat
    # lambda1 = lambda * w_1 / sigmak
    # lambda2 = lambda * w_2 / sigmak
    psimple = norm(u)^2 / (2 * tauk) + norm(transpose(X) * u)^2 / (2 * sigmak) - dot(u, X * betak - yk - Y)
    psimple -= sigmak * fusedMoreau(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak)
    psimple -= tauk * L2moreau(yk + u./tauk, 1/tauk)

    return psimple
end

# sparse group lasso gradPsi
function sparsegroupgradPsi(u::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T, group::Vector{Int64}, w_l) where T <: AbstractFloat
    # lambda1 = lambda * w_1 / sigmak
    # lambda2 = lambda * w_2 / sigmak
    pcomplex = Y
    pcomplex -= X * proxgroup(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak, group, w_l)
    pcomplex += proxL2norm(yk + u./tauk, 1/tauk)

    return pcomplex
end

# fused lasso gradPsi
function fusedgradPsi(u::Vector{T}, X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, yk::Vector{T}, sigmak::T, tauk::T) where T <: AbstractFloat
    # lambda1 = lambda * w_1 / sigmak
    # lambda2 = lambda * w_2 / sigmak
    pcomplex = Y
    pcomplex -= X * proxfused(betak - transpose(X) * u ./ sigmak, lambda * w_1 / sigmak, lambda * w_2 / sigmak)
    pcomplex += proxL2norm(yk + u./tauk, 1/tauk)

    return pcomplex
end