using SMPL;
using Flux;
using CUDA;
using BenchmarkTools;

home_dir = ENV["SENSEI_USERSPACE_SELF"]

smpl  = createSMPL(joinpath(home_dir,"projects/SMPL.jl/models/basicmodel_m_lbs_10_207_0_v1.0.0.npz"));

pose = rand(Float32,72);
betas = rand(Float32,10);

@btime smpl_lbs(smpl,betas,pose)



function smplj(betas,thetas)
    _,j = smpl_lbs(smpl,betas,thetas);

    return sum(j);
end

