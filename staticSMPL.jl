using SMPL;
using StaticTools;
using BenchmarkTools;


betas = rand(Float32,10)
pose = rand(Float32,72)
trans = rand(Float32,3)

@btime smpl_lbs($smpl_male,$betas,$pose,$trans);


betas = MallocVector{Float32}(undef,10)
pose = MallocVector{Float32}(undef,72)
trans = MallocVector{Float32}(undef,3)
betas .= 0.5
pose .= 0.5
trans .= 0.5

@btime smpl_lbs_static(smpl_static_male,betas,pose,trans);