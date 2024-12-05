using Revise;
using BenchmarkTools;
using SMPL;
using DataDeps;

# trans = ones(Float32,3);
# poses = ones(Float32,165);
# betas = ones(Float32, 400);

# smplx = create_smplx("./data/SMPLX_NEUTRAL.npz");

# @btime smpl_lbs($smplx,$betas,$poses,$trans);

# # smplxout = smpl_lbs(smplx,betas,poses,trans);


# trans = ones(Float32,3);
# poses = ones(Float32,72);
# betas = ones(Float32, 10);

# smpl = create_smpl(joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"));

# @btime smpl_lbs($smpl,$betas,$poses,$trans);
# # smplout = smpl_lbs(smpl,betas,poses,trans);


using SMPL
using DataDeps
using oneAPI
trans = oneArray(ones(Float32,3));
poses = oneArray(ones(Float32,72));
betas = oneArray(ones(Float32,10));

smpl = create_smpl(joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"));

for f in fieldnames(SMPL.SMPLdata)
    setfield!(smpl,f,oneArray(getfield(smpl,f)))
end

setfield!(smpl,:parents,Array(getfield(smpl,:parents)))

smpl_lbs(smpl,betas,poses,trans);



# ############ nested kernels ############
using oneAPI
using StaticArrays
N = 1
A = oneArray(ones(Float32,100,100,N))
B = oneArray(ones(Float32,100,100,N))
C = oneArray(ones(Float32,100,100,N))

function bmm!(C,A,B)
    i = get_global_id()
    a_view = @view A[:,:,i]
    a = @inbounds SArray{Tuple{100,100}}(a_view)
    b_view = @view B[:,:,i]
    b = @inbounds SArray{Tuple{100,100}}(b_view)
    c_view = @view C[:,:,i]
    c_view .= a * b
    return
end

@oneapi items=N bmm!(C, A, B)
synchronize()




# ################### CUDA BMM

using KernelAbstractions
GPU_PKG_NAME = "oneAPI"

if GPU_PKG_NAME == "CUDA"
    using CUDA, CUDAKernels
    const GPUMOD = CUDA
    const GpuArray = CuArray
    const GpuBackend = CUDADevice()
elseif GPU_PKG_NAME == "AMDGPU"
    using AMDGPU, ROCKernels
    const GPUMOD = AMDGPU
    const GpuArray = ROCArray
    const GpuBackend = ROCDevice()
elseif GPU_PKG_NAME == "oneAPI"
    using oneAPI
    const GPUMOD = oneAPI
    const GpuArray = oneArray
    const GpuBackend = CPU()
end


using StaticArrays
N = 1000
A = GpuArray(ones(Float32,100,100,100))
B = GpuArray(ones(Float32,100,100,100))
C = GpuArray(ones(Float32,100,100,100))

function bmm!(C,A,B)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:size(A,3)
        a_view = @view A[:,:,i]
        a = @inbounds SArray{Tuple{100,100}}(a_view)
        b_view = @view B[:,:,i]
        b = @inbounds SArray{Tuple{100,100}}(b_view)
        c_view = @view C[:,:,i]
        c_view .= a * b
    end
    return
end

numblocks = ceil(Int, N/256)

@cuda threads=256 blocks=numblocks bmm!(C,A,B)
