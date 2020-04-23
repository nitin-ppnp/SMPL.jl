module SMPLMod


using NPZ
using LinearAlgebra

struct SMPL
    v_template::Array{Float32,2};
    shapedirs::Array{Float32,3};
    posedirs::Array{Float32,2};
    J_regressor::Array{Float32,2};
    parents::Array{UInt32,1};
    lbs_weights::Array{Float32,2};
    betas::Array{Float32,1};
    thetas::Array{Float32,1};
    trans::Array{Float32,1}
    f::Array{UInt32,2};
end

function createSMPL(model_path::String)
    model = NPZ.npzread(model_path)
    parents = zeros(UInt32,24);
    parents[2:end] = model["parents"][2:end];
    f = reshape(model["f"]',(3,:));
    smpl = SMPL(model["v_template"],
                model["shapedirs"],
                model["posedirs"],
                model["J_regressor"],
                parents,
                model["lbs_weights"],
                zeros(Float32,10),
                zeros(Float32,72),
                zeros(Float32,3),
                f)        # python to cpp indexing

    return smpl
end

export createSMPL;

function smpl_lbs(betas::Array{Float32,1},pose::Array{Float32,1},smpl::SMPL,trans::Array{Float32,1}=zeros(Float32,3))
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
    v_shaped = smpl.v_template + reshape(reshape(smpl.shapedirs,(6890*3,:))*betas,(6890,3));
    
    J = smpl.J_regressor*v_shaped;
    
    if size(pose,1) == 24*3
        pose = reshape(pose,(3,24));
        rot_mats = cat([rodrigues(pose[:,i]) for i = 1:size(pose,2)]...,dims=3);
    elseif size(pose,1) == 24*3*3
        rot_mats = reshape(pose,(3,3,24));
    end
    
    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3);
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*smpl.posedirs,(3,:))';
    
    v_posed = pose_offsets + v_shaped;
    
    
    J_transformed, A = rigid_transform(rot_mats,J',smpl.parents.+1);
    
    T = reshape(reshape(A,(:,24))*smpl.lbs_weights',(4,4,:));
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,6890));
    v_homo = reduce(hcat,[T[:,:,i]*v_posed_homo[:,i] for i =1:6890]);
    verts = v_homo[1:3,:];
    return verts.+trans[:,[CartesianIndex()]], J_transformed.+trans[:,[CartesianIndex()]];
        
end


function rigid_transform(rot_mats,joints,parents)
    
    rel_joints = copy(joints);
    rel_joints[:,2:end] -= joints[:,parents[2:end]];
    
    transforms_mat = cat([vcat(hcat(rot_mats[:,:,i],rel_joints[:,i]),[0 0 0 1]) for i = 1:size(rot_mats)[3]]...,dims=3);
    
    transforms = zeros(Float32,size(transforms_mat));
    transforms[:,:,1] = transforms_mat[:,:,1];
    
    for i=2:size(parents)[1]
        transforms[:,:,i] = transforms[:,:,parents[i]] * transforms_mat[:,:,i];
    end
    
    posed_joints = transforms[1:3,4,:];

    
    joints_homo = vcat(joints,zeros(Float32,1,size(joints)[2]));
    
    init_bone = cat([hcat(zeros(Float32,4,3),transforms[:,:,i]*joints_homo[:,i]) for i=1:size(parents)[1]]...,dims=3);
    
    
    rel_transforms = transforms - init_bone;
        
    return posed_joints, rel_transforms;
    
end

function vertices2joints(J_regressor,vertices)
    
    return reshape(J_regressor*reshape(vertices,(6890,:)),(24,3,:));
    
    end

function rodrigues(rot_vec,eps=1.0f-8)
    
    angle = sqrt(sum((rot_vec.+eps).^2));
    rot_dir = rot_vec ./ angle;
    
    K = [0 -rot_dir[3] rot_dir[2] ;
        rot_dir[3] 0 -rot_dir[1] ;
        -rot_dir[2] rot_dir[1] 0];
    
    rot_mat = Matrix{Float32}(1.0I,3,3) + sin(angle)*K + (1-cos(angle))*K*K;
    
    return rot_mat;

end

Base.@ccallable function CSMPL_LBS(v_template::Array{Float32,2},
                    shapedirs::Array{Float32,3},
                    posedirs::Array{Float32,2},
                    J_regressor::Array{Float32,2},
                    parents::Array{UInt32,1},
                    lbs_weights::Array{Float32,2},
                    pose::Array{Float32,1},
                    betas::Array{Float32,1},
                    trans::Array{Float32,1},
                    v_ret::Array{Float32,2})::Array{Float32,2}
    """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    # @time begin
    # v_template = Array(reshape(reinterpret(Float32,v_templat),(6890,3)));
    # shapedirs = Array(reshape(reinterpret(Float32,shapedir),(6890,3,10)));
    # posedirs = Array(reshape(reinterpret(Float32,posedir),(207,20670)));
    # J_regressor = Array(reshape(reinterpret(Float32,J_regresso),(24,6890)));
    # lbs_weights = Array(reshape(reinterpret(Float32,lbs_weight),(6890,24)));
    # pose = Array(reshape(reinterpret(Float32,pos),(72)));
    # betas = Array(reshape(reinterpret(Float32,beta),(10)));
    # trans = Array(reshape(reinterpret(Float32,tran),(3)));
    # end

    # @show pose

    # @time begin
    rs = reshape(shapedirs,(6890*3,:));
    v_shaped = v_template + reshape(rs*betas,(6890,3));
    
    J = J_regressor*v_shaped;
    # end

    # @time begin
    # if size(pose,1) == 24*3
    pose = reshape(pose,(3,24));
    rot_mats = cat([rodrigues(pose[:,i]) for i = 1:size(pose,2)]...,dims=3);
    # elseif size(pose,1) == 24*3*3
    #     rot_mats = reshape(pose,(3,3,24));
    # end
    # end
    # @time begin
    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3);
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*posedirs,(3,:))';
    
    v_posed = pose_offsets + v_shaped;
    # end
    # @time begin
    J_transformed, A = rigid_transform(rot_mats,J',parents.+1);
    
    T = reshape(reshape(A,(:,24))*lbs_weights',(4,4,:));
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,6890));
    # end
    # @time begin
    v_homo = reduce(hcat,[T[:,:,i]*v_posed_homo[:,i] for i =1:6890]);
    # end
    verts = v_homo[1:3,:];
    v_final = verts.+trans[:,[CartesianIndex()]];
    
    # @time begin
    v_ret[:] = v_final[:];
    # end
    return v_ret; 
        
end


# Base.@ccallable function smpl_lbs2(v_template::Array{Float32,2},
#                     shapedirs::Array{Float32,3},
#                     posedirs::Array{Float32,2},
#                     J_regressor::Array{Float32,2},
#                     parents::Array{Int64,1},
#                     lbs_weights::Array{Float32,2})::Array{Float32,2}
#     """pose input (3x3)x24 : batch of 24 of 3x3 rotation matrices  """
    
#     return v_template';
        
# end

Base.@ccallable function CSMPL_v_template(v_template::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    v_template[:]   = smpl.v_template[:];
    return v_template;
end
Base.@ccallable function CSMPL_shapedirs(shapedirs::Array{Float32,3})::Array{Float32,3}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    shapedirs[:] = smpl.shapedirs[:];
    return shapedirs;
end
Base.@ccallable function CSMPL_posedirs(posedirs::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    posedirs[:] = smpl.posedirs[:];
    return posedirs;
end
Base.@ccallable function CSMPL_J_regressor(J_regressor::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    J_regressor[:] = smpl.J_regressor[:];
    return J_regressor;
end
Base.@ccallable function CSMPL_parents(parents::Array{UInt32,1})::Array{UInt32,1}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    parents[:] = smpl.parents[:];
    return parents;
end
Base.@ccallable function CSMPL_lbs_weights(lbs_weights::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    lbs_weights[:] = smpl.lbs_weights[:];
    return lbs_weights;
end
Base.@ccallable function CSMPL_f(f::Array{UInt32,2})::Array{UInt32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    f[:] = smpl.f[:];
    return f;
end
Base.@ccallable function test(test::Array{Float32,2})::Array{Float32,2}
    return reshape(reinterpret(Float32,test),(3,6890));
end
Base.@ccallable function CSMPL()::Any
    smpl = createSMPL(ENV["SMPLPATH"]);
    # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
    # print(size(verts))
    # @show joints
    return smpl;
end
# Base.@ccallable function CSMPL_v_template(v_template::Array{Float32,2})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     v_template[:]   = smpl.v_template[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_shapedirs(shapedirs::Array{Float32,3})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     shapedirs[:] = smpl.shapedirs[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_posedirs(posedirs::Array{Float32,2})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     posedirs[:] = smpl.posedirs[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_J_regressor(J_regressor::Array{Float32,2})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     J_regressor[:] = smpl.J_regressor[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_parents(parents::Array{UInt32,1})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     parents[:] = smpl.parents[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_lbs_weights(lbs_weights::Array{Float32,2})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     lbs_weights[:] = smpl.lbs_weights[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_f(f::Array{UInt32,2})::Cvoid
#     smpl = createSMPL("D:\\julia\\projects\\smpl\\smpl_new.npz");
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     f[:] = smpl.f[:];
#     return nothing;
# end
# Base.@ccallable function CSMPL_LBS(inp::Ptr{Cvoid})::Array{Float32,2}
#     smpl = Base.unsafe_convert(SMPL,inp);
#     verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     return verts;
# end


#g++ program.c -o main -lSMPLMod -ljulia -ID:\\julia\\julia-1.1.1\\include -ID:\\julia\\julia-1.1.1\\include\\julia -LD:\\julia\\julia-1.1.1\\lib -LD:\\julia\\projects\\smpl\\builddir
smpl = createSMPL(ENV["SMPLPATH"]);
CSMPL_LBS(smpl.v_template,smpl.shapedirs,smpl.posedirs,smpl.J_regressor,smpl.parents,smpl.lbs_weights,zeros(Float32,72),zeros(Float32,10),zeros(Float32,3),zeros(Float32,3,6890));
CSMPL_LBS(smpl.v_template,smpl.shapedirs,smpl.posedirs,smpl.J_regressor,smpl.parents,smpl.lbs_weights,zeros(Float32,72),zeros(Float32,10),zeros(Float32,3),zeros(Float32,3,6890));

end