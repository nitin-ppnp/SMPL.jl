using SMPL
using Test
using DataDeps
using NPZ

function test_smpl()
    datapath = joinpath(@__DIR__,"smpltest.npz")
    data = npzread(datapath);
    smpl = create_smpl_female();
    trans = data["trans"]
    betas = data["betas"]
    poses = data["poses"]
    out_vertices = data["out_vertices"]
    out_joints = data["out_joints"]
    out = smpl_lbs(smpl,betas,poses,trans);

    return maximum(abs.(out[1] - out_vertices')) < 1e-5 && maximum(abs.(out[2] - out_joints')) < 1e-5
end

function test_smplx()
    datapath = joinpath(@__DIR__,"smplxtest.npz")
    data = npzread(datapath);
    smplx = create_smplx_neutral();
    trans = data["trans"]
    betas = data["betas"]
    poses = data["poses"]
    out_vertices = data["out_vertices"]
    out_joints = data["out_joints"]
    out = smpl_lbs(smplx,betas,poses,trans);

    return maximum(abs.(out[1] - out_vertices')) < 1e-5 && maximum(abs.(out[2] - out_joints')) < 1e-5
end

@testset "SMPL" begin
    @test test_smpl()
end

@testset "SMPLX" begin
    @test test_smplx()
end
