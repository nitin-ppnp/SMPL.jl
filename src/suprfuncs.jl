using NPZ; npz=NPZ
using LinearAlgebra
using SparseArrays;
using SharedArrays;
using Distributed;
using TensorOperations;
using GLMakie;

struct SUPRdata
    v_template::Array{Float32,2}
    shapedirs::Array{Float32,3}
    posedirs::Array{Float32,3}
    J_regressor::SparseMatrixCSC{Float32, Int64}
    parents::Array{UInt32,1}
    lbs_weights::Array{Float32,2}
    f::Array{UInt32,2}
end

create_supr_female() = create_supr(joinpath(datadep"SUPR_models","SUPR_FEMALE.npz"));
create_supr_male() = create_supr(joinpath(datadep"SUPR_models","SUPR_MALE.npz"));
create_supr_neutral() = create_supr(joinpath(datadep"SUPR_models","SUPR_NEUTRAL.npz"));

function create_supr(model_path::String)
    """
    """
    model = NPZ.npzread(model_path);

    supr = SUPRdata(Float32.(model["v_template"]),
                    Float32.(model["shapedirs"]),
                    Float32.(model["posedirs"]),
                    sparse(Float32.(model["J_regressor"])),
                    UInt32.(model["kintree_table"][1,2:end].+1),
                    Float32.(model["weights"]),
                    model["f"].+1)        # python to julia indexing
    return supr
end

function quat_feat(theta)
    angle = norm(theta .+ 1f-8)
    normalized = theta ./ angle
    angle *= 0.5f0
    v_cos = cos(angle)
    v_sin = sin(angle)
    return vcat(v_sin * normalized, v_cos-1)
end

function rottrans2mat(rot, trans)
    return vcat(hcat(rot,trans),[0 0 0 1])
end

function so3_p_prod(rot, trans, point)
    return rot * point + trans
end

function so3_so3_prod(r1,t1,r2,t2)
    return r1*r2, r1*t2 + t1
end


function smpl_lbs(supr::SUPRdata,betas,pose=zeros(Float32,228),trans=zeros(Float32,3))
    """
    """
    
    nbetas = length(betas);
    njoints = 75;
    nverts = 10475;

    # v_shaped = reshape(reshape(supr.shapedirs,:,1:nbetas) * betas, size(supr.shapedirs,1),:) + supr.v_template;
    v_delta = zeros(Float32,nverts,3);
    @tensor v_delta[a,b] = supr.shapedirs[:,:,1:nbetas][a,b,c] * betas[c]
    v_shaped = supr.v_template + v_delta
    pad_v_shaped = [reshape(transpose(v_shaped),:);1];
    
    J = transpose(reshape(supr.J_regressor * pad_v_shaped,3,:));
    pose_quat = vcat([quat_feat(pose[3*(i-1)+1:3*i]) for i in axes(J,1)]...);
    
    R = vcat([reshape(rodrigues(pose[3*(i-1)+1:3*i]),1,3,3) for i in axes(J,1)]...);
    
    @tensor v_delta[a,b] = supr.posedirs[a,b,c]*pose_quat[c]
    v_posed = v_shaped + v_delta
    
    J_ = copy(J)
    J_[2:end,:] = J[2:end,:] - J[supr.parents,:]

    G = zeros(Float32,4,4,njoints);
    G[1:3,1:3,1] = R[1,:,:]
    G[1:3,4,1] = J_[1,:]
    for i = 2:75
        G[1:3,1:3,i], G[1:3,4,i] = so3_so3_prod(G[1:3,1:3,supr.parents[i-1]],
                                                G[1:3,4,supr.parents[i-1]],
                                                R[i,:,:],
                                                J_[i,:])
        G[4,4,i] = 1f0
    end

    for i = 1:size(G,3)
        G[1:3,4,i] -= G[1:3,1:3,i] * J[i,:]
    end
    
    @tensor T[a,b,c] := G[b,c,d] * supr.lbs_weights[a,d]

    v = zeros(Float32,nverts,3)
    for i in 1:nverts
        v[i,:] = so3_p_prod(@view(T[i,1:3,1:3]),trans,v_posed[i,:])
    end
    
    root_transform = rottrans2mat(R[1,:,:],J[1,:])
    results = [root_transform]
    
    
    for i in eachindex(supr.parents)
        transform_i = rottrans2mat(R[i+1,:,:],J[i+1,:] - J[supr.parents[i],:])
        curr_res = results[supr.parents[i]] * transform_i
        append!(results,[curr_res])
    end
    
    posed_joints = zeros(Float32,size(results,1),3);
    for i in eachindex(results)
        posed_joints[i,:] = results[i][1:3,4] + trans
    end
    
    output = Dict("vertices" => v,
                    "joints" => posed_joints,
                    "v_posed" => v_posed,
                    "v_shaped" => v_shaped,
                    "f" => supr.f) 

    return output
end

function smpl_lbs(supr::SUPRdata,betas::Array{Float32,2},pose::Array{Float32,1})
    output = Dict("vertices" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...), 
                    "v_posed" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...),
                    "v_shaped" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...),
                    "joints" => SharedArray{Float32}(size(betas,1),75,3),
                    "f" => supr.f) 
    
    Threads.@threads for i = 1:size(betas,1)
        out = supr_lbs(supr,betas[i,:],pose);
        output["vertices"][i,:,:] = out["vertices"]
        output["v_posed"][i,:,:] = out["v_posed"]
        output["v_shaped"][i,:,:] = out["v_shaped"]
        output["joints"][i,:,:] = out["joints"]
    end
    
    return output
end

function smpl_lbs(supr::SUPRdata,betas::Array{Float32,1},pose::Array{Float32,2})
    output = Dict("vertices" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...), 
                    "v_posed" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...),
                    "v_shaped" => SharedArray{Float32}(size(betas,1),size(supr.v_template)...),
                    "joints" => SharedArray{Float32}(size(betas,1),75,3),
                    "f" => supr.f) 
    
    Threads.@threads for i = 1:size(pose,1)
        out = smpl_lbs(supr,betas,pose[i,:]);
        output["vertices"][i,:,:] = out["vertices"]
        output["v_posed"][i,:,:] = out["v_posed"]
        output["v_shaped"][i,:,:] = out["v_shaped"]
        output["joints"][i,:,:] = out["joints"]
    end
    
    return output
end



#############################################################################

function viz_supr(supr::SUPRdata,betas::Array{Float32,1},pose::Array{Float32,1};kwargs...)
    verts = smpl_lbs(supr,betas,pose)["vertices"]
    f = Figure()
    scene = LScene(f[1,1],show_axis=false)
    mesh!(scene,verts',supr.f;kwargs...)
    cam = cameracontrols(scene)
    rotate_cam!(scene.scene,cam,(-0.95, -2.365, 0))
    return f
end