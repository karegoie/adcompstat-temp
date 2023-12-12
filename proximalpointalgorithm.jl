## proximal point algorithm
include("basics.jl")
include("proximal operator.jl")
include("derivation.jl")
include("genJaco.jl")
include("semismoothnewton.jl")

# sparsgegroup solver
function sparsegroupPPA(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, group::Vector{Int64}, w_l::Vector{T}) where T <: AbstractFloat

    # assertion
    (N, n) = size(X)
    @assert N == length(Y)
    @assert lambda > 0
    @assert w_1 >= 0
    @assert w_2 >= 0
    for i in 1:length(w_l)
        @assert w_l[i] >= 0
    end
    @assert length(group) == n
    @assert length(w_l) == maximum(group)
    

    # Initialized
    betak = zeros(n)
    yk = -Y
    sigmak = 1.0
    tauk = 1.0
    iter = 0
    betaklag = ones(n)
    start = Dates.now()
    time_elapsed = Dates.now() - start

    while sparsegroupterminate(X, Y, lambda, w_1, w_2, betak, betaklag, iter, time_elapsed, group, w_l)
        println("iter: ", iter)
        # step 1
        betaklag = copy(betak)
        (betak, yk) = sparsegroupSSN(X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk, group, w_l)

        # step 2
        sigmak = sigmak
        tauk = tauk
        iter += 1
        time_elapsed = Dates.now() - start

    end

    return (betak, yk)

end

# termination criterion of sparse group

function sparsegroupterminate(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, betaklag::Vector{T}, iter, time_elapsed, group, w_l) where T <: AbstractFloat

    # hyperparameters
    tol  = 1e-7
    maxiter = 20
    maxtime = Minute(5)

    # initialize
    flag = true
    residual = X * betak - Y

    if norm(residual) > tol
        barbeta = transpose(X) * residual ./ norm(residual)
        delta_kkt = norm(betak - proxgroup(betak - barbeta, lambda * w_1, lambda * w_2, group, w_l))
        if (delta_kkt < tol)
            println("Delta KKT:", delta_kkt)
            flag = false
        end
    end
    
    if norm(residual) < tol
        var_gap = norm(betak - betaklag) / (1 + norm(betak) + norm(betaklag))
        println("Relative successive change:", var_gap)
        flag = false
    end

    if iter > maxiter
        println("Terminate due to exceeding the max iteration")
        flag = false
    end

    if time_elapsed > maxtime
        println("Stopping local moves due to exceeding the time limit")
        flag = false
    end

    return flag
end

# fusedlasso solver
function fusedPPA(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T) where T <: AbstractFloat

    # assertion
    @assert lambda > 0
    @assert w_1 >= 0
    @assert w_2 >= 0
    
    
    (N, n) = size(X)
    @assert N == length(Y)

    # Initialized
    betak = zeros(n)
    yk = -Y
    sigmak = 1.0
    tauk = 1.0
    iter = 0
    betaklag = ones(n)
    start = Dates.now()
    time_elapsed = Dates.now() - start

    while fusedterminate(X, Y, lambda, w_1, w_2, betak, betaklag, iter, time_elapsed)
        println("iter: ", iter)
        # step 1
        betaklag = copy(betak)
        (betak, yk) = fusedSSN(X, Y, lambda, w_1, w_2, betak, yk, sigmak, tauk)

        # step 2
        sigmak = sigmak
        tauk = tauk
        iter += 1
        time_elapsed = Dates.now() - start

    end

    return (betak, yk)

end

# termination criterion of fused

function fusedterminate(X::AbstractMatrix{T}, Y::Vector{T}, lambda::T, w_1::T, w_2::T, betak::Vector{T}, betaklag::Vector{T}, iter, time_elapsed) where T <: AbstractFloat

    # hyperparameters
    tol  = 1e-7
    maxiter = 20
    maxtime = Minute(5)

    # initialize
    flag = true
    residual = X * betak - Y

    if norm(residual) > tol
        barbeta = transpose(X) * residual ./ norm(residual)
        delta_kkt = norm(betak - proxfused(betak - barbeta, lambda * w_1, lambda * w_2))
        if (delta_kkt < tol)
            println("Delta KKT:", delta_kkt)
            flag = false
        end
    end
    
    if norm(residual) < tol
        var_gap = norm(betak - betaklag) / (1 + norm(betak) + norm(betaklag))
        println("Relative successive change:", var_gap)
        flag = false
    end

    if iter > maxiter
        println("Terminate due to exceeding the max iteration")
        flag = false
    end

    if time_elapsed > maxtime
        println("Stopping local moves due to exceeding the time limit")
        flag = false
    end

    return flag
end