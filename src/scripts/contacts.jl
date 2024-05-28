using Revise;
using GLMakie;
using SMPL;
using NPZ;

function viz_skels(joints; bones=nothing,
    contacts=nothing,
    terrain_verts=nothing,
    terrain_faces=nothing,
    show_fig=false,
    tsleep=1 / 20,
    rec=false, recordFile="smplRecord.mkv", kwargs...)

    fig = Figure()
    l = LScene(fig[1,1:2],scenekw = (;limits=Rect3f(Vec3f(-10,-10,0),Vec3f(10,10,4))))
    status = Observable("Play")
    b = Button(fig[2,1], label = status)
    t = Slider(fig[2,2], range = 1:size(joints[1], 3), startvalue = 1);

    on(b.clicks) do n
        if status[] == "Play"
            status[] = "Pause"
            @async while (status[] == "Pause") & (t.value[] < size(joints[1], 3))
                set_close_to!(t, t.value[]+1)
                sleep(tsleep)
            end
        else
            status[] = "Play"
        end
    end

    jnts = []
    for j in joints
        push!(jnts,@lift j[:, :, $(t.value)])
    end
    if ~isnothing(contacts)
        clr = @lift contacts[:, $(t.value)]
    end
    display(fig)
    cmaps = [:Reds, :Greens]
    for (id,k) in enumerate(jnts[1:end])
        if ~isnothing(contacts)
            meshscatter!(l,k, color=clr, colormap=cmaps[id], markersize=0.05)
        else
            meshscatter!(l,k)
        end
    end
    fig
end

function viz_meshes(vertices,f; bones=nothing,
    contacts=nothing,
    terrain_verts=nothing,
    terrain_faces=nothing,
    show_fig=false,
    tsleep=1 / 20,
    rec=false, recordFile="smplRecord.mkv", kwargs...)

    fig = Figure()
    l = LScene(fig[1,1:4],scenekw = (;limits=Rect3f(Vec3f(-10,-10,0),Vec3f(10,10,4))))
    status = Observable("Play")
    b = Button(fig[2,1], label = status)
    t = Slider(fig[2,2], range = 1:size(vertices[1], 3), startvalue = 1);
    textbox = Textbox(fig[2, 3], placeholder = "Enter a string...", width=500)
    rec = Toggle(fig[2,4], active = false)

    on(rec.active) do n
        @async record(fig, "test.mkv"; framerate = 30) do io
            while rec.active[]
                sleep(tsleep)
                recordframe!(io) # record a new frame
            end
        end
    end

    on(b.clicks) do n
        if status[] == "Play"
            status[] = "Pause"
            @async while (status[] == "Pause") & (t.value[] < size(vertices[1], 3))
                set_close_to!(t, t.value[]+1)
                sleep(tsleep)
            end
        else
            status[] = "Play"
        end
    end

    verts = []
    for v in vertices
        push!(verts,@lift v[:, :, $(t.value)])
    end
    display(fig)
    for k in verts[1:end]
        mesh!(l,k,f)
    end
    fig
end

smplx_neutral = create_smplx_neutral();

data = npzread("/home/nitin/data/AMASS_SMPLX_NEUTRAL_contactsv2_JL/ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.npz");

betas = data["betas"][1:10];
poses = Matrix(data["poses"]');
contacts = Matrix(data["contact_probs"]');
trans = data["trans"][1,:];

out = [smpl_lbs(smplx_neutral,betas,poses[:,i],data["trans"][i,:]) for i in axes(poses,2)];
verts = stack(getindex.(out,1))
joints = stack(getindex.(out,2))

fk_verts, fk_joints = pivot_fk(smplx_neutral,
                            betas,
                            poses,
                            contacts,
                            data["trans"][1,:])


# viz = viz_skels([Float64.(joints[:,1:22,:])], contacts=contacts, show_fig=true, tsleep=1/100)
# viz = viz_skels([fk_joints[1:3,4,1:22,:]],  contacts=contacts, show_fig=true, tsleep=1/100)
# viz = viz_skels([fk_joints[1:3,4,1:22,:], Float64.(joints[:,1:22,:])],  contacts=contacts, show_fig=true, tsleep=1/100)
# viz2 = viz_meshes([verts, fk_verts], smplx_neutral.f ,show_fig=true, tsleep=1/100)


# retargeting experiments
betas_small = zeros(Float32,10);
betas_small[1] = -3;
out = [smpl_lbs(smplx_neutral,betas_small,poses[:,i],data["trans"][i,:]) for i in axes(poses,2)];
verts_norm_small_betas = stack(getindex.(out,1))
joints_norm_small_betas = stack(getindex.(out,2))

fk_verts_small, fk_joints_small = pivot_fk(smplx_neutral,
                            betas_small,
                            poses,
                            contacts,
                            data["trans"][1,:])

viz = viz_meshes([verts_norm_small_betas], smplx_neutral.f ,show_fig=true, tsleep=1/100)

pivot_j = argmax(contacts,dims=1)[1:end-1]
pivot_j_nextt = pivot_j.+CartesianIndex(0,1)
pivot_j_pos_orig = sqrt.(sum((joints[:,pivot_j_nextt] - 
                        joints[:,pivot_j]).^2,dims=1))
pivot_j_pos_smpl_fk_small = sqrt.(sum((joints_norm_small_betas[:,pivot_j_nextt] - 
                        joints_norm_small_betas[:,pivot_j]).^2,dims=1))
pivot_j_pos_pivot_fk_small = sqrt.(sum((fk_joints_small[1:3,4,pivot_j_nextt] - 
                        fk_joints_small[1:3,4,pivot_j]).^2,dims=1))

using Statistics

mean(pivot_j_pos_orig)
mean(pivot_j_pos_smpl_fk_small)
mean(pivot_j_pos_pivot_fk_small)
median(pivot_j_pos_orig)
median(pivot_j_pos_smpl_fk_small)
median(pivot_j_pos_pivot_fk_small)
