using GLMakie

function viz_skel(joints; bones=nothing,
    contacts=nothing,
    terrain_verts=nothing,
    terrain_faces=nothing,
    show=false,
    tsleep=1 / 20,
    rec=false, recordFile="smplRecord.mkv", kwargs...)

    joint = Observable(joints[:, :, 1])

    if ~isnothing(contacts)
        clr = Observable(contacts[:, 1])
    end
    msh = meshscatter(joint; kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    if show
        display(msh)
    end
    if rec
        record(msh, recordFile) do io
            for i in 1:size(joints, 3)
                joint[] = joints[:, :, i]
                if ~isnothing(contacts)
                    clr[] = contacts[:, i]
                end
                recordframe!(io)
                sleep(tsleep)
            end
        end
    else
        for i = 1:size(joints, 3)
            joint[] = joints[:, :, i]
            if ~isnothing(contacts)
                clr[] = contacts[:, i]
            end
            sleep(tsleep)
        end
    end
end

function viz_mesh(vertices,f; bones=nothing,
    contacts=nothing,
    terrain_verts=nothing,
    terrain_faces=nothing,
    show=false,
    tsleep=1 / 20,
    rec=false, recordFile="smplRecord.mkv", kwargs...)

    verts = Observable(vertices[:, :, 1])

    if ~isnothing(contacts)
        clr = Observable(contacts[:, 1])
    end
    msh = mesh(verts,f; kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    if show
        display(msh)
    end
    if rec
        record(msh, recordFile) do io
            for i in 1:size(joints, 3)
                verts[] = vertices[:, :, i]
                if ~isnothing(contacts)
                    clr[] = contacts[:, i]
                end
                recordframe!(io)
                sleep(tsleep)
            end
        end
    else
        for i = 1:size(vertices, 3)
            verts[] = vertices[:, :, i]
            if ~isnothing(contacts)
                clr[] = contacts[:, i]
            end
            sleep(tsleep)
        end
    end
end

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
                sleep(1/20)
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
    l = LScene(fig[1,1:2],scenekw = (;limits=Rect3f(Vec3f(-10,-10,0),Vec3f(10,10,4))))
    status = Observable("Play")
    b = Button(fig[2,1], label = status)
    t = Slider(fig[2,2], range = 1:size(vertices[1], 3), startvalue = 1);

    on(b.clicks) do n
        if status[] == "Play"
            status[] = "Pause"
            @async while (status[] == "Pause") & (t.value[] < size(vertices[1], 3))
                set_close_to!(t, t.value[]+1)
                sleep(1/20)
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

using NPZ;

# data = npzread("/home/nitin/nas/nitin/AMASS_SMPLX_NEUTRAL_contacts_v2/ACCAD/Female1General_c3d/A14_-_stand_to_skip_stageii.npz");
# data = npzread("/home/nitin/Downloads/A14_-_stand_to_skip_stageii.npz")
data = npzread("/home/nitin/Downloads/B10_-_walk_turn_left_(45)_stageii.npz")

# data = npzread("/home/nitin/nas/nitin/AMASS_SMPLX_NEUTRAL_contacts_v2/ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.npz");

probs = 100 .* Float64.(data["probs"]'[1:22,:]);
probs[probs.<0] .= 0;
contacts = zero.(probs);
contacts[argmax(probs,dims=1)] .= probs[argmax(probs,dims=1)]
# for i in 1:size(probs,2)
#     maximum(probs[:,i])
# argmax(probs,2)

using SMPL
# data2 = npzread("/home/nitin/nas/datasets/AMASS/AMASS_SMPLX_NEUTRAL_smplFormat/ACCAD/Female1General_c3d/A14_-_stand_to_skip_stageii.smpl")
data2 = npzread("/home/nitin/nas/datasets/AMASS/AMASS_SMPLX_NEUTRAL_smplFormat/ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.smpl")
poses = Array(Float32.(reshape(permutedims(data2["bodyPose"],(3,2,1)),(66,:))));
poses = cat(poses,zeros(Float32,55*3-66,size(poses,2));dims=1)
trans = Float32.(data2["bodyTranslation"]')
betas = zeros(Float32,10)

smplx_neutral = create_smplx_neutral()

fk_verts, fk_joints = pivot_fk(smplx_neutral,
                            betas,
                            poses,
                            Float32.(probs),
                            trans[:,1],
                            fps=data2["frameRate"]);

verts = [smpl_lbs(smplx_neutral,betas,poses[:,i],trans[:,i])["vertices"] for i in axes(poses,2)];
vertices = stack(verts)

viz = viz_meshes([Float64.(fk_verts), Float64.(vertices)], data["f"].+1, show_fig=true, tsleep=1/100)

# viz_skel(Float64.(data["contact_fk_vertices"]), contacts=probs, show=true, markersize=0.03,colormap=:Greens)

# viz = viz_meshes([Float64.(data["contact_fk_vertices"]), Float64.(data["vertices"])], data["f"].+1, show_fig=true, tsleep=1/100)
# viz = viz_skels([Float64.(data["contact_fk_joints"]), Float64.(data["joints"][:,1:22,:])], contacts=contacts, show_fig=true, tsleep=1/100)


# # plotting
# # vertices = [Float64.(data["contact_fk_vertices"]), Float64.(data["vertices"])]
# vertices = [Float64.(data["contact_fk_vertices"])]

# fig = Figure()
# l = LScene(fig[1,1:2],scenekw = (;limits=Rect3f(Vec3f(-10,-10,0),Vec3f(10,10,4))))
# status = Observable("Play")
# b = Button(fig[2,1], label = status)
# t = Slider(fig[2,2], range = 1:size(vertices[1], 3), startvalue = 1, snap = false);

# on(b.clicks) do n
#     if status[] == "Play"
#         status[] = "Pause"
#         @async while (status[] == "Pause") & (t.value[] < size(vertices[1], 3))
#         #     t.value[] = t.value[] + 1
#             set_close_to!(t, t.value[]+1)
#             sleep(1/20)
#         end
#     else
#         status[] = "Play"
#     end
# end

# verts = []
# for v in vertices
#     push!(verts,@lift v[:, :, $(t.value)])
# end

# display(fig)
# for k in verts[1:end]
#     mesh!(l,k,data["f"].+1)
# end

# # # for i in 1:size(vertices[1], 3)
# # #     for (j,v) in enumerate(vertices)
# # #         verts[j][] = v[:, :, i]
# # #     end
# # # end