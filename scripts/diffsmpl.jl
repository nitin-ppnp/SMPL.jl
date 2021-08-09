using SMPL;

smpl  = createSMPL("/home/nsaini/projects/SMPL/basicmodel_m_lbs_10_207_0_v1.0.0.npz");

function smplj(betas,thetas)
    _,j = smpl_lbs(smpl,betas,thetas);

    return sum(j);
end

