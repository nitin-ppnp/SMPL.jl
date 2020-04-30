
$JULIA_DIR/bin/julia src/pkg_install.jl

cd src

echo "creating precompile statements ..."
$JULIA_DIR/bin/julia --startup-file=no --trace-compile=smpl_precompile.jl csmpl.jl

echo "creating object file ..."
$JULIA_DIR/bin/julia --startup-file=no -J"$JULIA_DIR/lib/julia/sys.so" --output-o sysSMPL.o create_SMPLimage.jl

echo "creating shared library ..."
gcc -shared -o libcsmpl.so -Wl,--whole-archive sysSMPL.o -Wl,--no-whole-archive -L"$JULIA_DIR/lib" -ljulia


rm smpl_precompile.jl

rm sysSMPL.o

mkdir ../build

mv libcsmpl.so ../build

cd ..

cp $JULIA_DIR/lib/lib* build
cp $JULIA_DIR/lib/julia/* build
