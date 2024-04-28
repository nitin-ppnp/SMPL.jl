using DataDeps;
using HTTP;
using NPZ
using LinearAlgebra


function fetch_smpl_models(remote_filepath, local_directorypath)
    """
    """
    print("SMPL username: ")
    username = escape(readline())

    print("SMPL password: ")
    password = escape(readline())

    run(`
    wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=basicModel_f_lbs_10_207_0_v1.0.0.npz" -O $local_directorypath"/SMPL_FEMALE.npz" --no-check-certificate --continue`)

    run(`
    wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=basicmodel_m_lbs_10_207_0_v1.0.0.npz" -O $local_directorypath"/SMPL_MALE.npz" --no-check-certificate --continue`)

    return local_directorypath
end


function fetch_supr_models(remote_filepath, local_directorypath)
    """
    """
    print("SUPR username: ")
    username = escape(readline())

    print("SUPR password: ")
    password = escape(readline())

    run(`
    wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=supr&resume=1&sfile=male/body/SUPR_male.npz" -O $local_directorypath"/SUPR_MALE.npz" --no-check-certificate --continue`)

    run(`
    wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=supr&resume=1&sfile=female/body/SUPR_female.npz" -O $local_directorypath"/SUPR_FEMALE.npz" --no-check-certificate --continue`)

    run(`
    wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=supr&resume=1&sfile=generic/body/SUPR_neutral.npz" -O $local_directorypath"/SUPR_NEUTRAL.npz" --no-check-certificate --continue`)


    return local_directorypath
end


function fetch_smplx_models(remote_filepath, local_directorypath)
    """
    """
    # print("SMPLX username: ")
    # username = escape(readline())

    # print("SMPLX password: ")
    # password = escape(readline())

    # run(`
    # wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=basicModel_f_lbs_10_207_0_v1.0.0.npz" -O $local_directorypath"/SMPL_FEMALE.npz" --no-check-certificate --continue`)

    # run(`
    # wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=basicmodel_m_lbs_10_207_0_v1.0.0.npz" -O $local_directorypath"/SMPL_MALE.npz" --no-check-certificate --continue`)

    # return local_directorypath
end

function smpl_female_static(model_path = joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"))
    return createStaticSMPL(model_path)
end
function smpl_male_static(model_path = joinpath(datadep"SMPL_models","SMPL_MALE.npz"))
    return createStaticSMPL(model_path)
end


function __init__()
    """
    """
    register(DataDep("SMPL_models",
    "SMPL model files",
    "https://smpl.is.tue.mpg.de",
    Any;
    fetch_method = fetch_smpl_models,
    post_fetch_method = identity
    ))

    register(DataDep("SMPLX_models",
    "SMPLX model files",
    "https://smpl-x.is.tue.mpg.de",
    Any;
    fetch_method = fetch_smplx_models,
    post_fetch_method = identity
    ))

    register(DataDep("SUPR_models",
    "SUPR model files",
    "https://supr.is.tue.mpg.de",
    Any;
    fetch_method = fetch_supr_models,
    post_fetch_method = identity
    ))

end