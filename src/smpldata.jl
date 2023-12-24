using DataDeps;
using HTTP;


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

    global smpl_female = createSMPL(joinpath(datadep"SMPL_models","SMPL_FEMALE.npz"));
    global smpl_male = createSMPL(joinpath(datadep"SMPL_models","SMPL_MALE.npz"));

end