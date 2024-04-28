using Revise;
using BenchmarkTools;
using SMPL;
using DataDeps;

trans = ones(Float32,3);
poses = ones(Float32,165);
betas = ones(Float32, 400);

smplx = create_smplx("./data/SMPLX_NEUTRAL.npz");

@btime smpl_lbs($smplx,$betas,$poses,$trans);

# smplxout = smpl_lbs(smplx,betas,poses,trans);


trans = ones(Float32,3);
poses = ones(Float32,72);
betas = ones(Float32, 10);

smpl = create_smpl(joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"));

@btime smpl_lbs($smpl,$betas,$poses,$trans);
# smplout = smpl_lbs(smpl,betas,poses,trans);