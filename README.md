# CSMPL (compile SMPL.jl as shared library to use in C code)

## Requirements
1. Julia
2. MinGW (For Windows)

## Follow these steps to compile 
1. Get official julia (v1.4.1 used here) from https://julialang.org/downloads/oldreleases/ and open the julia executable to start Julia REPL. Execute the following commands


2. Get this repository and switch to branch CSMPL
```
git clone -b CSMPL https://github.com/nitin-ppnp/SMPL.jl
cd SMPL.jl
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
This will create a build directory named smplbuild, which contains all the needed shared library including **libcsmpl.so**.

5. Check if everything is working by building the sample program **program.cpp** included in the repo. First, set the env variable `CSMPL_LIB_PATH` to the absolute path of the newly built **libcsmpl.so**. Then, include the smplbuild directory in **LD_LIBRARY_PATH**. Afterwards, build the **program.cpp**.
```
export CSMPL_LIB_PATH={absolute path to the file libcsmpl.so}

export LD_LIBRARY_PATH={path of the smplbuild directory}:$LD_LIBRARY_PATH

g++ src/program.cpp -o main.out -lcsmpl -ljulia -L"./smplbuild" -I"$JULIA_DIR/include/julia/"

./main.out
```

**On Windows**

GCC is needed to build the shared library. We use MinGW shell on windows. Get MinGW shell and execute the following commands.

3. Set the environment variable `JULIA_DIR` as path to julia directory
```
set JULIA_DIR={path to julia directory}
```

4. The csmpl library expects the path to SMPL model file in ".npz" format. Download the models from https://smpl.is.tue.mpg.de/ Set the environment variable `SMPLPATH` with the path of downloaded SMPL model.
```
set SMPLPATH={path to SMPL model .npz file}
```

4. run the script `buildSMPL.bat`
```
buildSMPL.bat
```
This will create a build directory named smplbuild, which contains all the needed shared library including **csmpl.dll**.

5. Check if everything is working by building the sample program **program.cpp** included in the repo. First, set the env variable `CSMPL_LIB_PATH` to the absolute path of the newly built **csmpl.dll**. Afterwards, build the **program.cpp**.
```
set CSMPL_LIB_PATH={absolute path to the file csmpl.dll}

g++ src\program.cpp -o main.exe -lcsmpl -ljulia -L".\\smplbuild" -I"%JULIA_DIR%\\include\\julia"

move main.exe smplbuild             

smplbuild\main.exe
```
