ECHO OFF

%JULIA_DIR%\bin\julia src\pkg_install.jl

cd src

echo "creating precompile statements ..."
%JULIA_DIR%\bin\julia --startup-file=no --trace-compile=smpl_precompile.jl csmpl.jl

echo "creating object file ..."
%JULIA_DIR%\bin\julia --startup-file=no -J"%JULIA_DIR%\\lib\\julia\\sys.dll" --output-o sysSMPL.o create_SMPLimage.jl

echo "creating shared library ..."
gcc -shared -o csmpl.dll -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"%JULIA_DIR%\\lib" -ljulia -Wl,--export-all-symbols


del smpl_precompile.jl

del sysSMPL.o

mkdir ..\smplbuild

move csmpl.dll ..\smplbuild

cd ..

copy %JULIA_DIR%\bin\*.dll smplbuild
copy %JULIA_DIR%\lib\julia\* smplbuild