module SMPLViz

using GLMakie
using SMPL

function viz_supr(supr::SMPL.SUPRdata,betas::Array{Float32,1},pose::Array{Float32,1};kwargs...)
    verts = smpl_lbs(supr,betas,pose)["vertices"]
    f = Figure()
    scene = LScene(f[1,1],show_axis=false)
    mesh!(scene,verts',supr.f;kwargs...)
    cam = cameracontrols(scene)
    rotate_cam!(scene.scene,cam,(-0.95, -2.365, 0))
    return f
end

function SMPL.viz_smpl(smpl::SMPL.SMPLdata,betas::Array{Float32,1},pose::Array{Float32,1};kwargs...)
    verts = smpl_lbs(smpl,betas,pose)["vertices"]
    scene = GLMakie.mesh(verts',smpl.f;kwargs...)
    return scene
end

function viz_smpl(smpl::SMPL.SMPLdata,betas::Array{Float32,2},pose::Array{Float32,1};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs2(smpl,betas,pose,smpl)
    
    vert = Node(verts[:,:,1]')
    msh = GLMakie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        GLMakie.record(msh,recordFile,1:size(verts,3)) do i
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    else
        for i=1:size(verts,3)
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    end
end

function viz_smpl(smpl::SMPL.SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs2(smpl,betas,pose)
    
    vert = Node(verts[:,:,1]')
    msh = GLMakie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        GLMakie.record(msh,recordFile,1:size(verts,3)) do i
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    else
        for i=1:size(verts,3)
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    end
end

function viz_smpl(smpl::SMPL.SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2},trans::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs2(smpl,betas,pose,trans)
    vert = Node(verts[:,:,1]')
    msh = GLMakie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        GLMakie.record(msh,recordFile,1:size(verts,3)) do i
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    else
        for i=1:size(verts,3)
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    end
end

function viz_smpl(smpl::SMPL.SMPLdata,betas::Array{Float32,2},pose::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs2(smpl,betas,pose)
    
    vert = Node(verts[:,:,1]')
    msh = GLMakie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        GLMakie.record(msh,recordFile,1:size(verts,3)) do i
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    else
        for i=1:size(verts,3)
            vert[] = verts[:,:,i]'
            sleep(tsleep)
        end
    end
end

function viz_skel(joints; bones=nothing,
    contacts=nothing,
    terrain_verts=nothing,
    terrain_faces=nothing,
    show=false,
    tsleep=1 / 20,
    rec=true, recordFile="smplRecord.mkv", kwargs...)

    joint = Observable(joints[:, :, 1])

    clr = Observable(contacts[:, 1])
    msh = meshscatter(joint, color=clr; kwargs...)
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
                clr[] = contacts[:, i]
                recordframe!(io)
                sleep(tsleep)
            end
        end
    else
        for i = 1:size(joints, 3)
            joint[] = joints[:, :, i]
            clr[] = contacts[:, i]
            sleep(tsleep)
        end
    end
end
    
end