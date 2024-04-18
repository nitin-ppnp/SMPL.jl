module SMPL

include("smpldata.jl")
include("utils.jl")
include("smplfuncs.jl");
include("smplxfuncs.jl");
export create_smpl_female;
export create_smpl_male;
export create_smpl_neutral
export create_smplx_female;
export create_smplx_male;
export create_smplx_neutral;
export create_smpl;
export create_smplx;
export smpl_lbs;
export pivot_fk;

export viz_smpl;

end # module
