using SMPL;
using GLMakie;
using Colors;
using NPZ;

function viz_smpl(smpl::SMPL.SMPLdata,betas::AbstractArray{Float32,1},
                    pose::AbstractArray{Float32,2},
                    trans::AbstractArray{Float32,2},
                    conts::AbstractArray{Float32,2};
                    tsleep=1/10,
                    record=false,
                    recordFile="smplRecord.mp4",kwargs...)
    verts,J = smpl_lbs(smpl,betas,pose,trans)
    t = Node(1);
    vert = @lift verts[:,:,$t]
    # probs = prob_remap.(conts[4,:,:])
    joint_spheres = [@lift Sphere(Point3f0(J[1:3,j,$t]),conts[j,$t]/10) for j =1:24]
    cmap = colormap("RdBu")
    clrs = [@lift cmap[101 - Int(ceil(conts[j,$t]*99) + 1)] for j = 1:24]
    # cont_radius = [@lift conts[4,:,$t]]
    msh = mesh(vert,smpl.f,shininess=5.0f0;kwargs...)
    for i =1:24
        mesh!(joint_spheres[i],color=clrs[i])
    end
    xlims!(msh.axis.scene,(-10,10))
    ylims!(msh.axis.scene,(-10,10))
    # msh[Axis][:showaxis] = (false,false,false);
    # msh[Axis][:showgrid] = (false,true,false);
    # msh[Axis][:showticks] = (false,false,false);
    
    if record
        display(msh)
        GLMakie.record(msh,recordFile,1:size(verts,3)) do i
            t[] = i
            update_cam!(msh.axis.scene,J[:,1,i]+Float32.([4,0,0]),J[:,1,i])
            sleep(tsleep)
        end
    else
        display(msh)
        for i=1:size(verts,3)
            t[] = i
            update_cam!(msh.axis.scene,J[:,1,i]+Float32.([4,0,0]),J[:,1,i])
            sleep(tsleep)
        end
    end
end

da = npzread("/Users/natinsaini/projects/nmg_viz/nmg_logs/Trial_0309_with_conts_mse_loss/train_nmg_13606_00019_19_conts_loss_weight=10,conts_speed_loss_weight=0.01,gt_conts_for_speed_loss=True,kl_annealing_cycle_epochs=_2021-09-05_04-44-33/epoch=999-step=251999val_cont_rand_morelen_latent_sample.npz");

smpl = createSMPL("/Users/natinsaini/projects/SMPL.jl/models/basicmodel_m_lbs_10_207_0_v1.0.0.npz");

pose_body = permutedims(cat(da["root_orient"],da["pose_body"],zeros(Float32,size(da["pose_body"])[1:2]...,6);dims=(3)),[1,3,2]);
trans = permutedims(da["trans"],[1,3,2]);
contacts = permutedims(cat(da["contacts"],zeros(Float32,size(da["contacts"])[1:2]...,2);dims=(3)),[1,3,2]);

id = 1
viz_smpl(smpl,zeros(Float32,10),pose_body[id,:,:],trans[id,:,:],contacts[id,:,:])
verts,joints = smpl_lbs(smpl,zeros(Float32,10),pose_body[1,:,:],trans[1,:,:]);