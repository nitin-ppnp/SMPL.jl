# SMPL
Julia implementation of Skinned Multi-Person Linear model (SMPL). It is based on the pytorch implementation https://github.com/vchoutas/smplx

![](resources/smpl.gif)

## To use SMPL as a shared library, checkout the CSMPL branch


## Using the package
```julia
]add https://github.com/nitin-ppnp/SMPL.jl
```

- run the following code to visualize the zero pose and shape. Importing `SMPL` for the first time will initiate the model files download.  
```julia
using SMPL;

# get vertices and 3D joints
vertices,joints = smpl_lbs(smpl.smpl_female,zeros(Float32,10),zeros(Float32,72));
# visualize zero pose and shape
viz_smpl(smpl,zeros(Float32,10),zeros(Float32,72),color=:turquoise)
```


# Benchmarking
- Current smpl_lbs impl: 2.617 ms