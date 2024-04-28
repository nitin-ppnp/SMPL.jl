module SMPL

include("smpldata.jl")
include("utils.jl")
include("smplfuncs.jl");
include("smplxfuncs.jl");
include("suprfuncs.jl");
export create_smpl_female;
export create_smpl_male;
export create_smpl_neutral
export create_smplx_female;
export create_smplx_male;
export create_smplx_neutral;
export create_smpl;
export create_smplx;
export create_supr_female;
export create_supr_male;
export create_supr_neutral;
export create_supr;
export smpl_lbs;
export pivot_fk;

end
