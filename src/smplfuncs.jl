using NPZ; npz=NPZ
using LinearAlgebra
using SharedArrays;
using Distributed;
using Flux;
using Zygote;

struct SMPLdata
    v_template::Array{Float32,2}
    shapedirs::Array{Float32,3}
    posedirs::Array{Float32,2}
    J_regressor::Array{Float32,2}
    parents::Array{UInt32,1}
    lbs_weights::Array{Float32,2}
    f::Array{UInt32,2}
end



function createSMPL(model_path)
    """
    """
    
    model = NPZ.npzread(model_path);

    smpl = SMPLdata(Float32.(model["v_template"]),
                Float32.(model["shapedirs"]),
                Float32.(model["posedirs"]),
                Float32.(model["J_regressor"]),
                UInt32.(model["kintree_table"][1,:]),
                Float32.(model["weights"]),
                model["f"].+1)        # python to julia indexing
    return smpl
end



function smpl_lbs(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,1},trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    v_shaped = smpl.v_template + reshape(reshape(smpl.shapedirs,(6890*3,:))*betas,(6890,3))
    
    J = smpl.J_regressor*v_shaped
    
    
    if size(pose,1) == 24*3
        pose = reshape(pose,(3,24))
        rot_mats = cat([rodrigues(pose[:,i]) for i = 1:size(pose,2)]...,dims=3)
    elseif size(pose,1) == 24*3*3
        rot_mats = reshape(pose,(3,3,24))
    end
    
    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3)
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*smpl.posedirs,(3,:))'
    
    v_posed = pose_offsets + v_shaped
    
    
    J_transformed, A = rigid_transform(rot_mats,J',smpl.parents.+1)
    
    T = reshape(reshape(A,(:,24))*smpl.lbs_weights',(4,4,:))
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,6890))
    # temp = [T[:,:,i]*v_posed_homo[:,i] for i =1:6890];
    # v_homo = reduce(hcat,temp);
    v_homo = dropdims(batched_mul(T,v_posed_homo[:,[CartesianIndex()],:]);dims=2);
    verts = v_homo[1:3,:]
    return verts.+trans[:,[CartesianIndex()]], J_transformed.+trans[:,[CartesianIndex()]]
        
end



function smpl_lbs(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,1})
    verts = SharedArray{Float32}(size(smpl.v_template)[end:-1:1]...,size(betas,2));
    joints = SharedArray{Float32}(3,24,size(betas,2));
    
    @sync @distributed for i = 1:size(pose,2)
        verts[:,:,i], joints[:,:,i] = smpl_lbs(smpl,betas[:,i],pose);
    end
    
    return verts,joints 
end

function smpl_lbs(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2})
    
    verts = SharedArray{Float32}(size(smpl.v_template)[end:-1:1]...,size(pose,2));
    joints = SharedArray{Float32}(3,24,size(pose,2));
    
    @sync @distributed for i = 1:size(pose,2)
        verts[:,:,i], joints[:,:,i] = smpl_lbs(smpl,betas,pose[:,i]);
    end
    
    return verts,joints 
end

function smpl_lbs(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2},trans::Array{Float32,2})
    
    verts = SharedArray{Float32}(size(smpl.v_template)[end:-1:1]...,size(pose,2));
    joints = SharedArray{Float32}(3,24,size(pose,2));
    
    @sync @distributed for i = 1:size(pose,2)
        verts[:,:,i], joints[:,:,i] = smpl_lbs(smpl,betas,pose[:,i],trans[:,i]);
    end
    
    return verts,joints 
end
    
function smpl_lbs(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,2},trans::Array{Float32,2})
    verts = SharedArray{Float32}(size(smpl.v_template)[end:-1:1]...,size(pose,2));
    joints = SharedArray{Float32}(3,24,size(pose,2));
    
    @sync @distributed for i = 1:size(pose,2)
        verts[:,:,i], joints[:,:,i] = smpl_lbs(smpl,betas[:,i],pose[:,i],trans[:,i]);
    end
    
    return verts,joints 
end

function smpl_lbs2(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,2},trans::Array{Float32,2})
    out = [smpl_lbs(smpl,betas[:,i],pose[:,i],trans[:,i]) for i = 1:size(pose,2)]
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints 
end

function smpl_lbs2(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2})
    pose_lbs(t) = smpl_lbs(smpl,betas,t)
    out = mapslices(pose_lbs,pose,dims=1)
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints
end

function smpl_lbs2(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,1})
    shape_lbs(t) = smpl_lbs(smpl,t,pose)
    out = mapslices(shape_lbs,betas,dims=1)
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints
end


function smpl_lbs2(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2},trans::Array{Float32,2})
    out = [smpl_lbs(smpl,betas,pose[:,i],trans[:,i]) for i = 1:size(pose,2)]
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints 
end

# function smpl_lbs(betas,pose,smpl,repr="aa")
#     if repr == "aa"
#         return smpl_lbs(betas,reshape(pose,3*24,:),smpl)
#     elseif repr == "mat"
#         return smpl_lbs(betas,reshape(pose,3*3*24,:),smpl)
#     end
# end

# function get_transforms(transforms_mat,parents)
#     transforms = bufferfrom(zeros(Float32,size(transforms_mat)))

#     for i=2:size(parents)[1]
#         transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i]
#     end
# end


function rigid_transform(rot_mats,joints,parents)
    
    rel_joints1 = joints[:,1]
    # rel_joints[:,2:end] -= joints[:,parents[2:end]]
    rel_joints = hcat(rel_joints1,joints[:,2:end]-joints[:,parents[2:end]])
    
    transforms_mat = cat([vcat(hcat(rot_mats[:,:,i],rel_joints[:,i]),[0 0 0 1]) for i = 1:size(rot_mats)[3]]...,dims=3)
    
    transf = Dict()
    transf[1] = transforms_mat[:,:,1]
    
    # for i=2:size(parents)[1]
    #     transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i]
    # end
    buf = Zygote.Buffer(zeros(Float32,size(transforms_mat)),size(transforms_mat))
    for i=2:size(parents)[1]
        transf[i] = transf[parents[i]] * transforms_mat[:,:,i]
        buf[:,:,i] = transf[i]
    end

    transforms = copy(buf)
    # transforms = cat(transf...,dims=3)

    
    posed_joints = transforms[1:3,4,:]

    
    joints_homo = vcat(joints,zeros(Float32,1,size(joints)[2]))

    init_bone = hcat(zeros(Float32,4,3,24),batched_mul(transforms,joints_homo[:,[CartesianIndex()],:]))
    
    # temp = [hcat(zeros(Float32,4,3),transforms[:,:,i]*joints_homo[:,i]) for i=1:size(parents)[1]]
    # init_bone = cat(temp...,dims=3)
    
    
    rel_transforms = transforms - init_bone

        
    return posed_joints, rel_transforms
    
end



function vertices2joints(J_regressor,vertices)
    
    return reshape(J_regressor*reshape(vertices,(6890,:)),(24,3,:))
    
    end


function rodrigues(rot_vec,eps=1.0f-8)
    
    angle = sqrt(sum((rot_vec.+eps).^2))
    rot_dir = rot_vec ./ angle
    
    K = [0 -rot_dir[3] rot_dir[2] ;
        rot_dir[3] 0 -rot_dir[1] ;
        -rot_dir[2] rot_dir[1] 0]
    
    rot_mat = Matrix{Float32}(1.0I,3,3) + sin(angle)*K + (1-cos(angle))*K*K
    
    return rot_mat

end

# smpl = createSMPL("/is/sg2/nsaini/aerial-pose-tracker_rel/aeropose/torchSMPL/smpl_m.npz");

# function zygtest(pose::Array{Float32,1})
    
    
#     a,b = lbs(zeros(Float32,10),pose,smpl);
    
#     return sum(a)
    
# end
    

########################## Viz ###########################
using Makie

function viz_smpl(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,1};kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose)
    scene = Makie.mesh(verts',smpl.f;kwargs...)
    return scene
end

function viz_smpl(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,1};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose,smpl)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        Makie.record(msh,recordFile,1:size(verts,3)) do i
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

function viz_smpl(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        Makie.record(msh,recordFile,1:size(verts,3)) do i
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

function viz_smpl(smpl::SMPLdata,betas::Array{Float32,1},pose::Array{Float32,2},trans::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose,trans)
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        Makie.record(msh,recordFile,1:size(verts,3)) do i
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

function viz_smpl(smpl::SMPLdata,betas::Array{Float32,2},pose::Array{Float32,2};tsleep=1/10,record=false,recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    display(msh)
    if record
        Makie.record(msh,recordFile,1:size(verts,3)) do i
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

# function viz_smpl(betas,pose,smpl::SMPLdata ; repr::String="aa")
#     verts,J = smpl_lbs(betas,pose,smpl,repr)
    
#     vert = Node(verts[:,:,1]')
#     msh = Makie.mesh(vert,smpl.f)
#     display(msh)
#     for i=1:size(verts,3)
#         push!(vert,verts[:,:,i]')
#         sleep(1/10)
#     end
# end