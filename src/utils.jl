using oneAPI
using LinearAlgebra

function rodrigues(rot_vec,eps=1.0f-8)
    
    angle = sqrt.(sum((rot_vec.+eps).^2))
    rot_dir = rot_vec ./ angle
    
    # K = [0 -rot_dir[3] rot_dir[2] ;
    #     rot_dir[3] 0 -rot_dir[1] ;
    #     -rot_dir[2] rot_dir[1] 0]
    
    K = similar(rot_vec,3,3)
    oneAPI.@allowscalar K[1,2] = -rot_dir[3]
    oneAPI.@allowscalar K[1,3] = rot_dir[2]
    oneAPI.@allowscalar K[2,1] = rot_dir[3]
    oneAPI.@allowscalar K[2,3] = -rot_dir[1]
    oneAPI.@allowscalar K[3,1] = -rot_dir[2]
    oneAPI.@allowscalar K[3,2] = rot_dir[1]
    

    rot_mat = oneArray{Float32}(I,3,3) + sin(angle)*K + (1-cos(angle))*K*K
    
    return rot_mat

end
