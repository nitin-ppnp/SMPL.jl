# CSMPL (compile SMPL.jl as shared library to use in C code)

## Follow these steps to compile 
1. Get official julia (v1.1.1) from https://julialang.org/downloads/oldreleases/ and open the julia executable to start Julia REPL. Execute the following commands
```julia
import Pkg;
Pkg.add("LinearAlgebra");
Pkg.add("NPZ");
```


2. Get this repository and switch to branch CSMPL
```
git clone -b CSMPL https://github.com/nitin-ppnp/SMPL.jl
cd SMPL
```

**On Ubuntu**

3. Set the environment variable `JULIA_DIR` as path to julia directory
```
export JULIA_DIR={path to julia directory}
```

4. The csmpl library expects the path to SMPL model file in ".npz" format. Download the models from https://smpl.is.tue.mpg.de/ Set the environment variable `SMPLPATH` with the path of downloaded SMPL model.
```
export SMPLPATH={path to SMPL model .npz file}
```

4. run the script `buildSMPL.sh`
```
sh buildSMPL.sh
```
This will create a build directory which contains all the needed shared library including **libcsmpl.so**.

5. Check if everything is working by building the sample program **program.cpp** included in the repo. First, set the env variable `CSMPL_LIB_PATH` to the absolute path of the newly built **libcsmpl.so**. Then, include the build directory in **LD_LIBRARY_PATH**. Afterwards build the **program.cpp**.
```
export CSMPL_LIB_PATH={absolute path to the file libcsmpl.so}

export LD_LIBRARY_PATH={path of the build directory}:$LD_LIBRARY_PATH

g++ src/program.cpp -o main.out -lcsmpl -ljulia -L"./build" -I"$JULIA_DIR/include/julia/"

./main.out
```

**On Windows**

4. In the following commands, `{julia}` is the path to julia executable, usually it is `{JuliaDirectory}\\bin\\julia`. `{julia library directory}` is the path to julia library directory which is `{JuliaDirectory}\\bin`. Use MinGW to execute following commands.
```
{julia} --startup-file=no --trace-compile=smpl_precompile.jl src/csmpl.jl

{julia} --startup-file=no --output-o sysSMPL.o create_SMPLimage.jl

gcc -shared -o csmpl.dll -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"{julia library directory}" -ljulia -Wl,--export-all-symbols
```

5. Create a new directory `build` and copy the resultant `csmpl.dll` to this directory. Additionally copy the content of `JuliaDirectory}\\bin` to `build`.

The resultant build directory now contain the csmpl shared library and other libraries it needs. The user need to link against the csmpl shared library while compiling C/C++ code. The repository contains an example C++ file which uses csmpl shared library for loading SMPL model, performing the linear blend skinning and print the resultant vertices. Following are the instructions to build the example C++ file and run it.

```
g++ program.cpp -o build/main -lcsmpl -ljulia -I{JuliaDirectory}/include -I{JuliaDirectory}/include/julia -Ljulia/lib -L./build
```