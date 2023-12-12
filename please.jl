## Test by examples

include("basics.jl")
include("proximal operator.jl")
include("derivation.jl")
include("genJaco.jl")
include("semismoothnewton.jl")
include("proximalpointalgorithm.jl")

X = [1 2 3 4 5; 2 3.0 2.0 5.0 4.0; 4.0 6.0 5.0 2.0 8.0]
Y = [1; 2; 3.0]
lambda = 0.5
w_1 = 0.7
w_2 = 0.3
group = [1, 2, 2, 3, 3]
w_l = [0.2, 0.3, 0.5]

print(sparsegroupPPA(X, Y, lambda, w_1, w_2, group, w_l))
# print(fusedPPA(X, Y, lambda, w_1, w_2))