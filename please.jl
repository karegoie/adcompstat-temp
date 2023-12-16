## Test by examples

include("basics.jl")
include("proximal_operator.jl")
include("derivation.jl")
include("genJaco.jl")
include("semismoothnewton.jl")
include("proximalpointalgorithm.jl")

using DelimitedFiles


X = [1 2 3 4 5; 2.8 3.4 2.5 5.6 4.2; 4.0 6.0 5.0 2.0 8.0; 1.4 2.7 3.1 4.5 5.5]
Y = [1; 2; 3.0; 4.0]
lambda = 1.0
group = [1; 1; 2; 3; 3; 2]
w_1 = 0.7
w_2 = 0.3
w_l = [0.2; 0.3; 0.5]


#=
Y = float.((readdlm("GSE3330_inbredmouse.txt")[297,2:end]))
X = Matrix(float.(transpose(readdlm("GSE3330_inbredmouse.txt")[[2:296;298:end],2:end])))

(N, n) = size(X)

lambda = 1.0
w_1 = 0.7
w_2 = 0.3

#=
# odd group = 1, even group = 2
group = zeros(Int64, n)
for l in 1:n
    if l % 2 == 0
        group[l] = 1
    else
        group[l] = 2
    end 
end

w_l = [0.5; 0.5]
=#
=#
#print(sparsegroupPPA(X, Y, lambda, w_1, w_2, group, w_l))
print(fusedPPA(X, Y, lambda, w_1, w_2))