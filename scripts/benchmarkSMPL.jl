# using Distributed
using SMPL;
using BenchmarkTools;


smpl = createSMPL("models/basicModel_f_lbs_10_207_0_v1.0.0.npz");

batch_sizes = [1,2,3,4,5,6,7,8,9,10,20,30,40]

b = zeros(size(batch_sizes)[1])

for i in 1:size(batch_sizes)[1]
    batch_size = batch_sizes[i]
    pose = rand(Float32,72,batch_size);
    betas = rand(Float32,10,batch_size);
    trans = rand(Float32,3,batch_size);

    bmark = @benchmark smpl_lbs($smpl,$betas,$pose,$trans)
    # println(median(bmark))

    b[i] = median(bmark).time

    println(batch_size)
end