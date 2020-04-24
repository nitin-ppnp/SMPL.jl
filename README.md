# CSMPL (compile SMPL.jl as shared library to use in C code)

## Follow these steps to compile 
1. Get official julia (v1.1.1) from https://julialang.org/downloads/oldreleases/

2. Get this repository and switch to branch CSMPL
```
git clone -b CSMPL https://github.com/nitin-ppnp/SMPL.jl
cd SMPL
```
**On Linux**

3. In the following commands, `{julia}` is the path to julia executable, usually it is `{JuliaDirectory}/bin/julia`. `{julia library directory}` is the path to julia library directory which is `{JuliaDirectory}/lib`
```
{julia} --startup-file=no --trace-compile=smpl_precompile.jl src/csmpl.jl

{julia} --startup-file=no --output-o sysSMPL.o create_SMPLimage.jl

gcc -shared -o libcsmpl.so -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"{julia library directory}" -ljulia
```
4. Create a new directory `build` and copy the resultant `libcsmpl.so` to this directory. Additionally copy the content of `JuliaDirectory}/lib` to `build`.

**On Windows**

3. In the following commands, `{julia}` is the path to julia executable, usually it is `{JuliaDirectory}\\bin\\julia`. `{julia library directory}` is the path to julia library directory which is `{JuliaDirectory}\\bin`. Use MinGW to execute following commands.
```
{julia} --startup-file=no --trace-compile=smpl_precompile.jl src/csmpl.jl

{julia} --startup-file=no --output-o sysSMPL.o create_SMPLimage.jl

gcc -shared -o csmpl.dll -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"{julia library directory}" -ljulia -Wl,--export-all-symbols
```

4. Create a new directory `build` and copy the resultant `csmpl.dll` to this directory. Additionally copy the content of `JuliaDirectory}\\bin` to `build`.


The resultant build directory now contain the csmpl shared library and other libraries it needs. The user need to link against the csmpl shared library while compiling C/C++ code. The repository contains an example C++ file which uses csmpl shared library for loading SMPL model, performing the linear blend skinning and print the resultant vertices. Following are the instructions to build the example C++ file and run it.

