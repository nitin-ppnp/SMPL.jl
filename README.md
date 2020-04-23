# CSMPL (compile SMPL.jl as shared library to use in C code)

## Follow these steps to compile 
1. Get official julia (v1.1.1) from https://julialang.org/

2. Get this repository and switch to branch CSMPL
```
git clone -b CSMPL https://github.com/nitin-ppnp/SMPL.jl
cd SMPL
```
**On Linux**

3. `{julia}` is the path to julia executable, usually it is `{JuliaDirectory}/bin/julia`
```
{julia} --startup-file=no --trace-compile=smpl_precompile.jl src/csmpl.jl

{julia} --startup-file=no --output-o sysSMPL.o create_SMPLimage.jl
```


4. `{julia library directory}` is the path to julia library directory which is `{JuliaDirectory}/lib`
```
gcc -shared -o libcsmpl.so -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"{julia library directory}" -ljulia **-Wl,--export-all-symbols**
```
**On Windows**

3. `{julia}` is the path to julia executable, usually it is `{JuliaDirectory}/bin/julia`
```
{julia} --startup-file=no --trace-compile=smpl_precompile.jl src/csmpl.jl

{julia} --startup-file=no --output-o sysSMPL.o create_SMPLimage.jl
```
4. `{julia library directory}` is the path to julia library directory which is `{JuliaDirectory}\\bin`
```
gcc -shared -o csmpl.dll -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"{julia library directory}" -ljulia -Wl,--export-all-symbols
```



# Install
## Python Dependencies
Since `PyCall` is used for loading the official SMPL model .pkl file, the python environment should have following dependencies installed. 
1. scipy
2. chumpy (`pip install chumpy`)
3. Pickle

## Using the package
```julia
]add https://github.com/nitin-ppnp/SMPL.jl
```
- Download the SMPL model files from the SMPL website https://smpl.is.tue.mpg.de/

- run the following code to visualize the zero pose and shape.
```julia
using SMPL;
# create SMPL model
smpl = createSMPL("path/to/the/SMPL/model/.pkl/file");
# get vertices and 3D joints
vertices,joints = smpl_lbs(smpl,zeros(Float32,10),zeros(Float32,72));
# visualize zero pose and shape
viz_smpl(smpl,zeros(Float32,10),zeros(Float32,72),color=:turquoise)
```