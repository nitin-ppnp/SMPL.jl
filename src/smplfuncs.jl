using NPZ; npz=NPZ
using DataDeps;

struct SMPLdata
    v_template::Array{Float32,2}
    shapedirs::Array{Float32,2}
    posedirs::Array{Float32,2}
    J_regressor::Array{Float32,2}
    parents::Array{UInt32,1}
    lbs_weights::Array{Float32,2}
    f::Array{UInt32,2}
end



create_smpl_female() = create_smpl(joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"))
create_smpl_male() = create_smpl(joinpath(datadep"SMPL_models","SMPL_MALE.npz"))
create_smpl_neutral() = create_smpl(joinpath(datadep"SMPL_models","SMPL_NEUTRAL.npz"))

function create_smpl(model_path)
    """
    """
    model = NPZ.npzread(model_path);
    model["shapedirs"] = reshape(model["shapedirs"],(6890*3,:))
    model["posedirs"] = reshape(model["posedirs"],6890*3,:)'
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
    
    v_shaped = smpl.v_template + reshape((@view smpl.shapedirs[:,1:length(betas)]) * betas,(6890,3))
    
    J = smpl.J_regressor*v_shaped
    
    pose_view = reshape(pose,(3,24))
    rot_mats = zeros(Float32,3,3,24)
    @inbounds for i in axes(rot_mats,3)
        rot_mats[:,:,i] = rodrigues(pose_view[:,i])
    end
    
    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3)
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*smpl.posedirs,(:,3))
    
    v_posed = pose_offsets + v_shaped
    
    
    J_transformed, A = rigid_transform_smpl(rot_mats,J',smpl.parents.+1)
    
    T = reshape(reshape(A,(:,24))*smpl.lbs_weights',(4,4,:))
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,6890))

    v_homo = zeros(Float32,4,6890)
    @inbounds @simd for i = 1:6890
        v_homo[:,i] = T[:,:,i] * v_posed_homo[:,i]
    end

    # v_homo = zeros(Float32,4,6890)
    # loop_einsum!(EinCode((('i','j', 'k'),('j','k')),('i','k')),(T,v_posed_homo),v_homo)

    verts = @views v_homo[1:3,:] .+ trans[:,[CartesianIndex()]]

    output = Dict("vertices" => verts, 
                    "joints" => J_transformed[1:3,4,:].+trans[:,[CartesianIndex()]],
                    "v_posed" => v_posed,
                    "v_shaped" => v_shaped,
                    "J_transformed" => J_transformed,
                    "f" => smpl.f)
    return output
        
end


function rigid_transform_smpl(rot_mats,joints,parents)
    
    rel_joints = copy(joints)
    rel_joints[:,2:end] -= joints[:,parents[2:end]]

    transforms_mat = zeros(Float32,4,4,24)

    for i = 1:24
        transforms_mat[1:3,1:3,i] = rot_mats[:,:,i]
        transforms_mat[1:3,4,i] = rel_joints[:,i]
        transforms_mat[4,4,i] = 1
    end

    # transforms_mat = cat([vcat(hcat(rot_mats[:,:,i],rel_joints[:,i]),[0 0 0 1]) for i = 1:size(rot_mats)[3]]...,dims=3)
    
    transforms = zeros(Float32,size(transforms_mat))
    transforms[:,:,1] = transforms_mat[:,:,1]
    
    for i=2:24
        transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i]
    end
    
    posed_joints = copy(transforms)

    
    joints_homo = vcat(joints,zeros(Float32,1,size(joints,2)))

    init_bone = zeros(Float32,4,24)
    for i = 1:24
        init_bone[:,i] = transforms[:,:,i]*joints_homo[:,i]
    end

    transforms[:,4,:] = transforms[:,4,:] - init_bone

    return posed_joints, transforms
    
end



function vertices2joints_smpl(J_regressor,vertices)
    
    return reshape(J_regressor*reshape(vertices,(6890,:)),(24,3,:))
    
end


function pivot_fk(smpl::SMPLdata,betas::Array{Float32,1},poses::Array{Float32,2},contacts::Array{Float32,2},trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    verts = zeros(3,6890,size(poses,2));
    joints = zeros(4,4,24,size(poses,2));
    ot = smpl_lbs(smpl,betas,poses[:,1]);
    v_ot, j_ot = ot["vertices"], ot["J_transformed"] 
    verts[:,:,1] = v_ot .+ trans[:,[CartesianIndex()]]
    joints[:,:,:,1] = j_ot
    joints[1:3,4,:,1] .+= trans[:,[CartesianIndex()]]

    # contact joints
    contact_joints = argmax(contacts,dims=1)
    for (idx,cj) in enumerate(contact_joints[1:end-1])
        v_ot_fut, _, j_ot_fut = smpl_lbs(smpl,betas,poses[:,idx+1])
        
        # all the joints relative to the contact joint
        cj_inv_pose = inv(j_ot_fut[:,:,cj[1]])
        rel_joints_ot_fut = stack([cj_inv_pose * j_ot_fut[:,:,i] for i in axes(j_ot_fut,3)])
        
        # joint_ot relative to past joint ot
        joint_rel_past = inv(j_ot[:,:,cj[1]]) * j_ot_fut[:,:,cj[1]]

        # check free fall
        if false
            vel = (out_joints[i,j,:3,3] - out_joints[i-1,j,:3,3]) * fps + \
                torch.tensor(g).float().to(out_joints.device)/fps
            # vel = torch.matmul(torch.linalg.inv(out_joints[i,j]),out_joints[i-1,j])[:3,3] * fps + \
            #     torch.tensor(g).float().to(out_joints.device)/fps
            joint_rel_past[:3,3] = torch.einsum("ak,k->a",torch.linalg.inv(joints_ot[i,j])[:3,:3],vel)/fps
        else
            joint_rel_past[1:3,4] .= 0
        end

        # rotate past joint with joint_ot relative to past joint_ot
        rotated_fut_joint = joints[:,:,cj[1],idx] * joint_rel_past
        # place future pose_ot at past joint
        joints[:,:,:,idx+1] = stack([rotated_fut_joint * rel_joints_ot_fut[:,:,i] for i in axes(rel_joints_ot_fut,3)])
        
        # verts wrt contact joint ot
        verts_wrt_cj_ot = cj_inv_pose[1:3,1:3] * v_ot_fut .+ cj_inv_pose[1:3,4]
        verts[:,:,idx+1] = joints[1:3,1:3,cj[1],idx+1] * verts_wrt_cj_ot .+ joints[1:3, 4, cj[1], idx+1]

        # 
        j_ot = j_ot_fut
    end


    return verts, joints
        
end