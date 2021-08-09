# using Distributed

# res = Vector()

# for nproc in [1]
#     println("entering main loop")
#     # println("number of workers = ", nworkers())
#     print(pwd())
#     include("benchmarkSMPL.jl")
#     append!(res,b)
#     # addprocs(1)
# end

# function pose_jacobian(f, x)
#     y = f(x)
#     n = length(y)
#     m = length(x)
#     T = eltype(y)
#     j = Array{T, 2}(undef, n, m)
#     for i in 1:n
#         j[i, :] .= gradient(x -> f(x)[i], x)[1]
#     end
#     return j
# end

using SMPL

pose = rand(Float32,72)
betas = rand(Float32,10)
trans = rand(Float32,3)
smpl = createSMPL("/home/nsaini/projects/SMPL.jl/models/basicModel_f_lbs_10_207_0_v1.0.0.npz");

smpl_lbs(smpl,betas,pose,trans)