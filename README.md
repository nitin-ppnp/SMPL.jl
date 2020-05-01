# SMPL
Julia implementation of Skinned Multi-Person Linear model (SMPL). It is based on the pytorch implementation https://github.com/vchoutas/smplx

## To use SMPL as a shared library, checkout the CSMPL branch


## Using the package
```julia
]add https://github.com/nitin-ppnp/SMPL.jl
```
- Download the SMPL model files (.npz format) from the SMPL website https://smpl.is.tue.mpg.de/

- run the following code to visualize the zero pose and shape.
```julia
using SMPL;
# create SMPL model
smpl = createSMPL("path/to/the/SMPL/model/.npz/file");
# get vertices and 3D joints
vertices,joints = smpl_lbs(smpl,zeros(Float32,10),zeros(Float32,72));
# visualize zero pose and shape
viz_smpl(smpl,zeros(Float32,10),zeros(Float32,72),color=:turquoise)
```