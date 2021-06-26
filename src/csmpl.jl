module SMPLMod


using NPZ;
using LinearAlgebra;

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
    parents[2:end] = model["kintree_table"][1,2:end];
    f = reshape(model["f"]',(3,:));
    smpl = SMPL(model["v_template"],
                model["shapedirs"],
                model["posedirs"],
                model["J_regressor"],
                parents,
                model["weights"],
                zeros(Float32,10),
                zeros(Float32,72),
                zeros(Float32,3),
                f.+1)        # python to cpp indexing

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

function _lg_cross(A::AbstractArray, B::AbstractArray)
    if !(size(A, 1) == size(B, 1) == 3)
        throw(
            DimensionMismatch(
                "cross product is only defined for AbstractArray of dimension 3 at dims 1",
            ),
        )
    end
    a1, a2, a3 = A[1, :], A[2, :], A[3, :]
    b1, b2, b3 = B[1, :], B[2, :], B[3, :]
    return vcat(
        reshape.(
            [(a2 .* b3) - (a3 .* b2), (a3 .* b1) - (a1 .* b3), (a1 .* b2) - (a2 .* b1)],
            1,
            :,
        )...,
    )
end

_norm(A::AbstractArray; dims::Int = 2) = sqrt.(sum(A .^ 2; dims = dims))

function _normalize(A::AbstractArray{T,2}; eps::Number = 1e-6, dims::Int = 2) where {T}
    eps = T.(eps)
    norm = max.(_norm(A; dims = dims), eps)
    return (A ./ norm)
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

    rs = reshape(shapedirs,(6890*3,:));
    v_shaped = v_template + reshape(rs*betas,(6890,3));
    
    J = J_regressor*v_shaped;

    pose = reshape(pose,(3,24));
    rot_mats = cat([rodrigues(pose[:,i]) for i = 1:size(pose,2)]...,dims=3);

    pose_feature = permutedims(rot_mats[:,:,2:end],[2,1,3]) .- Matrix{Float32}(1I,3,3);
    
    pose_offsets = reshape(reshape(pose_feature,(1,:))*posedirs,(3,:))';
    
    v_posed = pose_offsets + v_shaped;

    J_transformed, A = rigid_transform(rot_mats,J',parents.+1);
    
    T = reshape(reshape(A,(:,24))*lbs_weights',(4,4,:));
    
    v_posed_homo = vcat(v_posed',ones(Float32,1,6890));

    v_homo = reduce(hcat,[T[:,:,i]*v_posed_homo[:,i] for i =1:6890]);
    # end
    verts = v_homo[1:3,:];
    v_final = verts.+trans[:,[CartesianIndex()]];
    
    v_ret[:] = v_final[:];
    # end
    return v_ret; 
        
end


Base.@ccallable function CSMPL_get_normals(verts::Array{Float32,2},faces::Array{UInt32,2}, normals::Array{Float32,2})::Array{Float32,2}
    vert_faces = verts[:,faces];
    
    vertex_normals = zeros(typeof(verts[1]),size(verts));
    vertex_normals[:, faces[1, :]] += _lg_cross(
        vert_faces[:, 2, :] - vert_faces[:, 1, :],
        vert_faces[:, 3, :] - vert_faces[:, 1, :],
    );
    vertex_normals[:, faces[2, :]] += _lg_cross(
        vert_faces[:, 3, :] - vert_faces[:, 2, :],
        vert_faces[:, 1, :] - vert_faces[:, 2, :],
    );
    vertex_normals[:, faces[3, :]] += _lg_cross(
        vert_faces[:, 1, :] - vert_faces[:, 3, :],
        vert_faces[:, 2, :] - vert_faces[:, 3, :],
    );

    normals[:] = _normalize(copy(vertex_normals), dims = 1);
    return normals;
end


Base.@ccallable function CSMPL_v_template(v_template::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    v_template[:]   = smpl.v_template[:];
    return v_template;
end
Base.@ccallable function CSMPL_shapedirs(shapedirs::Array{Float32,3})::Array{Float32,3}
    smpl = createSMPL(ENV["SMPLPATH"]);
    shapedirs[:] = smpl.shapedirs[:];
    return shapedirs;
end
Base.@ccallable function CSMPL_posedirs(posedirs::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    posedirs[:] = smpl.posedirs[:];
    return posedirs;
end
Base.@ccallable function CSMPL_J_regressor(J_regressor::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    J_regressor[:] = smpl.J_regressor[:];
    return J_regressor;
end
Base.@ccallable function CSMPL_parents(parents::Array{UInt32,1})::Array{UInt32,1}
    smpl = createSMPL(ENV["SMPLPATH"]);
    parents[:] = smpl.parents[:];
    return parents;
end
Base.@ccallable function CSMPL_lbs_weights(lbs_weights::Array{Float32,2})::Array{Float32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    lbs_weights[:] = smpl.lbs_weights[:];
    return lbs_weights;
end
Base.@ccallable function CSMPL_f(f::Array{UInt32,2})::Array{UInt32,2}
    smpl = createSMPL(ENV["SMPLPATH"]);
    f[:] = smpl.f[:];
    return f;
end

# Base.@ccallable function CSMPL()::Any
#     smpl = createSMPL(ENV["SMPLPATH"]);
#     # verts,_ = smpl_lbs(smpl.betas,smpl.thetas,smpl);
#     # print(size(verts))
#     # @show joints
#     return smpl;
# end


#g++ program.c -o main -lSMPLMod -ljulia -ID:\\julia\\julia-1.1.1\\include -ID:\\julia\\julia-1.1.1\\include\\julia -LD:\\julia\\julia-1.1.1\\lib -LD:\\julia\\projects\\smpl\\builddir
smpl = createSMPL(ENV["SMPLPATH"]);
CSMPL_LBS(smpl.v_template,smpl.shapedirs,smpl.posedirs,smpl.J_regressor,smpl.parents,smpl.lbs_weights,zeros(Float32,72),zeros(Float32,10),zeros(Float32,3),zeros(Float32,3,6890));
CSMPL_LBS(smpl.v_template,smpl.shapedirs,smpl.posedirs,smpl.J_regressor,smpl.parents,smpl.lbs_weights,zeros(Float32,72),zeros(Float32,10),zeros(Float32,3),zeros(Float32,3,6890));
CSMPL_get_normals(copy(smpl.v_template'),smpl.f,zeros(Float32,3,6890));
CSMPL_get_normals(copy(smpl.v_template'),smpl.f,zeros(Float32,3,6890));

end