# SMPL
Julia implementation of Skinned Multi-Person Linear model (SMPL). It is based on the pytorch implementation https://github.com/vchoutas/smplx

![](resources/smpl.gif)

> **Note:** To use SMPL as a shared library, checkout the `CSMPL` branch


## Using the package
```julia
]add https://github.com/nitin-ppnp/SMPL.jl
```

- run the following code to visualize the zero pose and shape. Creating the `SMPL/SMPLX/SUPR` structs for the first time will initiate the model files download. To download the models, you need register at the model websites, which are the following
1. For SMPL: [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
2. For SMPLX: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
3. For SUPR: [https://supr.is.tue.mpg.de/](https://supr.is.tue.mpg.de/)

```julia
using SMPL;

# Create SMPL/SMPLX/SUPR data structs
# first time execution will ask the credentials for the respective model website
smpl = create_smpl_neutral();
smplx = create_smplx_neutral();
supr = create_supr_neutral();

# Define betas and poses arrays
betas = zeros(Float32, 10);
poses = zeros(Float32, 72);

# Call smpl_lbs function with SMPL data struct
output_smpl = smpl_lbs(smpl, betas, poses);

# Define betas and poses arrays
betas = zeros(Float32, 10);
poses = zeros(Float32, 165);

# Call smpl_lbs function with SMPLX data struct
output_smplx = smpl_lbs(smplx, betas, poses);

# Define betas and poses arrays
betas = zeros(Float32, 10);
poses = zeros(Float32, 228);

# Call smpl_lbs function with SUPR data struct
output_supr = smpl_lbs(supr, betas, poses);

# Access the vertices from the output dict
vertices_smpl = output_smpl["vertices"];
vertices_smplx = output_smplx["vertices"];
vertices_supr = output_supr["vertices"];
```


# Benchmarking
- Current smpl_lbs impl: 2.617 ms