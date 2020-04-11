using NPZ; npz=NPZ
using LinearAlgebra
using Debugger;

struct SMPLdata
    v_template::Array{Float32,2}
    shapedirs::Array{Float32,3}
    posedirs::Array{Float32,2}
    J_regressor::Array{Float32,2}
    parents::Array{Int64,1}
    lbs_weights::Array{Float32,2}
    f::Array{UInt32,2}
end


function createSMPL(model_path)
    model = npz.npzread(model_path)
#     smpl = SMPL(v_template=model["v_template"],
#                 shapedirs=model["shapedirs"],
#                 posedirs=model["posedirs"],
#                 J_regressor=model["J_regressor"],
#                 parents=model["parents"],
#                 lbs_weights=model["lbs_weights"])
    smpl = SMPLdata(model["v_template"],
                model["shapedirs"],
                model["posedirs"],
                model["J_regressor"],
                model["parents"],
                model["lbs_weights"],
                model["f"].+1)        # python to julia indexing
    return smpl
end

# function blend_shapes(betas,shape_disps)
    
#     return reshape(reshape(smpl.shapedirs,(6890*3,:))*betas,(6890,3,:)) 

# end


function smpl_lbs(betas::Array{Float32,1},pose::Array{Float32,1},smpl::SMPLdata,trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    v_shaped = smpl.v_template + reshape(reshape(smpl.shapedirs,(6890*3,:))*betas,(6890,3))
    
    J = smpl.J_regressor*v_shaped
    @bp
    
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
    v_homo = reduce(hcat,[T[:,:,i]*v_posed_homo[:,i] for i =1:6890])
    verts = v_homo[1:3,:]
    return verts.+trans[:,[CartesianIndex()]], J_transformed.+trans[:,[CartesianIndex()]]
        
end


function smpl_lbs(betas::Array{Float32,2},pose::Array{Float32,1},smpl::SMPLdata)
    shape_lbs(t) = smpl_lbs(t,pose,smpl)
    out = mapslices(shape_lbs,betas,dims=1)
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints
end

function smpl_lbs(betas::Array{Float32,1},pose::Array{Float32,2},smpl::SMPLdata)
    pose_lbs(t) = smpl_lbs(betas,t,smpl)
    out = mapslices(pose_lbs,pose,dims=1)
    o1 = [t[1] for t in out]
    o2 = [t[2] for t in out]
    verts = reshape(reduce(hcat,o1),3,6890,:)
    joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints
end
    
function smpl_lbs2(betas::Array{Float32,1},pose::Array{Float32,2},smpl::SMPLdata)
    bsize = size(pose,2)
    verts = zeros(Float32,3,6890,bsize)
    joints = zeros(Float32,3,24,bsize)
    for i=1:bsize
        verts[:,:,i],joints[:,:,i] = smpl_lbs(betas,pose[:,i],smpl)
    end
#     pose_lbs(t) = smpl_lbs(betas,t,smpl)
#     out = mapslices(pose_lbs,pose,dims=1)
#     o1 = [t[1] for t in out]
#     o2 = [t[2] for t in out]
#     verts = reshape(reduce(hcat,o1),3,6890,:)
#     joints = reshape(reduce(hcat,o2),3,24,:)
    return verts,joints
end

function smpl_lbs(betas::Array{Float32,2},pose::Array{Float32,2},smpl::SMPLdata)
    out = [smpl_lbs(betas[:,i],pose[:,i],smpl) for i = 1:size(pose,2)]
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


function rigid_transform(rot_mats,joints,parents)
    
    rel_joints = copy(joints)
    rel_joints[:,2:end] -= joints[:,parents[2:end]]
    
    transforms_mat = cat([vcat(hcat(rot_mats[:,:,i],rel_joints[:,i]),[0 0 0 1]) for i = 1:size(rot_mats)[3]]...,dims=3)
    
    transforms = zeros(Float32,size(transforms_mat))
    transforms[:,:,1] = transforms_mat[:,:,1]
    
    for i=2:size(parents)[1]
        transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i]
    end
    
    posed_joints = transforms[1:3,4,:]

    
    joints_homo = vcat(joints,zeros(Float32,1,size(joints)[2]))
    
    init_bone = cat([hcat(zeros(Float32,4,3),transforms[:,:,i]*joints_homo[:,i]) for i=1:size(parents)[1]]...,dims=3)
    
    
    rel_transforms = transforms - init_bone
        
    return posed_joints, rel_transforms
    
end

# function lbs(betas,pose,smpl::SMPLdata)
    
#     batch_size = max(size(betas,ndims(betas)),size(pose,ndims(pose)))
    
#     b_shapes = blend_shapes(betas,smpl.shapedirs)
    
#     v_shaped = [smpl.v_template[i,j]+b_shapes[i,j,k] for i=1:size(b_shapes,1),j=1:size(b_shapes,2),k=1:size(b_shapes,3)]
        
#     J = vertices2joints(smpl.J_regressor,v_shaped)
    
#     rot_mats = cat([rodrigues(pose[:,i]) for i = 1:size(pose,2)]...,3)
    
#    pose_feature =  
        
# end


function vertices2joints(J_regressor,vertices)
    
    return reshape(J_regressor*reshape(vertices,(6890,:)),(24,3,:))
    
    end

# function batch_rodrigues(rot_vecs,eps=1e-8)
    
#     batch_size = size(rot_vecs,ndims(rot_vecs))
    
#     angle = sqrt.(sum((rot_vecs+eps).^2,dims=1))
#     angle = reshape(angle,(1,ndims(angle)))
#     rot_dir = rot_vecs ./ angle
    
#     co = cos.(angle)
#     si = sin.(angle)
    
#     rx,ry,rz = rot_dir[1,:],rot_dir[2,:],rot_dir[3,:]
#     k = 
    


# end

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

function viz_smpl(betas::Array{Float32,1},pose::Array{Float32,1},smpl::SMPLdata;kwargs...)
    verts,J = smpl_lbs(betas,pose,smpl)
    scene = Makie.mesh(verts',smpl.f;kwargs...)
    return scene
end

function viz_smpl(betas::Array{Float32,2},pose::Array{Float32,1},smpl::SMPLdata;tsleep=1/10,kwargs...)
    verts,J = smpl_lbs(betas,pose,smpl)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    display(msh)
    for i=1:size(verts,3)
        push!(vert,verts[:,:,i]')
        sleep(tsleep)
    end
end

function viz_smpl(betas::Array{Float32,1},pose::Array{Float32,2},smpl::SMPLdata;tsleep=1/10,kwargs...)
    verts,J = smpl_lbs(betas,pose,smpl)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    display(msh)
    for i=1:size(verts,3)
        push!(vert,verts[:,:,i]')
        sleep(tsleep)
    end
end

function viz_smpl(betas::Array{Float32,2},pose::Array{Float32,2},smpl::SMPLdata;tsleep=1/10,kwargs...)
    verts,J = smpl_lbs(betas,pose,smpl)
    
    vert = Node(verts[:,:,1]')
    msh = Makie.mesh(vert,smpl.f;kwargs...)
    display(msh)
    for i=1:size(verts,3)
        push!(vert,verts[:,:,i]')
        sleep(tsleep)
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