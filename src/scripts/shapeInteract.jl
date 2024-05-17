using Observables;
using SMPL;
using GLMakie;

smplx = create_smplx_neutral();

f = Figure()

scene = LScene(f[1,1],show_axis=false)


β_1 = Slider(f[2,1], range = -5:0.01:5, startvalue = 0);
β_2 = Slider(f[3,1], range = -5:0.01:5, startvalue = 0);
β_3 = Slider(f[4,1], range = -5:0.01:5, startvalue = 0);
β_4 = Slider(f[5,1], range = -5:0.01:5, startvalue = 0);
β_5 = Slider(f[6,1], range = -5:0.01:5, startvalue = 0);
β_6 = Slider(f[7,1], range = -5:0.01:5, startvalue = 0);
β_7 = Slider(f[8,1], range = -5:0.01:5, startvalue = 0);
β_8 = Slider(f[9,1], range = -5:0.01:5, startvalue = 0);
β_9 = Slider(f[10,1], range = -5:0.01:5, startvalue = 0);
β_10 = Slider(f[11,1], range = -5:0.01:5, startvalue = 0);


β = @lift(Float32.([$(β_1.value),
            $(β_2.value),
            $(β_3.value),
            $(β_4.value),
            $(β_5.value),
            $(β_6.value),
            $(β_7.value),
            $(β_8.value),
            $(β_9.value),
            $(β_10.value)]));


out = @lift(smpl_lbs(smplx,$β, zeros(Float32,55*3))["vertices"]);

mesh!(scene,out,smplx.f,color = :Turquoise)

cam = cameracontrols(scene)

rotate_cam!(scene.scene,cam,(-0.95, -2.365, 0))


display(f)