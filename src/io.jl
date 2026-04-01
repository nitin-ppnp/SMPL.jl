# io.jl — Model loading for SMPL, SMPLX, and SUPR.
#
# Public API:
#   create_smpl(path)         -> BodyModel{Float32, Matrix{Float32}}
#   create_smplx(path)        -> BodyModel{Float32, Matrix{Float32}}
#   create_supr(path)         -> SUPRModel{Float32, Matrix{Float32}}
#
# Convenience wrappers (trigger DataDeps download on first call):
#   create_smpl_female/male/neutral()
#   create_smplx_female/male/neutral()
#   create_supr_female/male/neutral()
#
# DataDeps registration:
#   __init__()  — registers SMPL_models, SMPLX_models, SUPR_models
#                 with credential-based HTTP download from is.tue.mpg.de
#
# Credentials:
#   Copy credentials.toml.example → credentials.toml and fill in your details.
#   If credentials.toml is absent or empty, an interactive prompt is shown instead.
#
# All `create_*` functions return CPU models with concrete Matrix{Float32} fields.
# Array conversions (to Float32, 1-indexed) happen once at load time.
# Move to GPU with: Adapt.adapt(CuArray, model)  (requires CUDA.jl loaded)

using NPZ
using DataDeps
using HTTP
using TOML
using LinearAlgebra: I
using SparseArrays
using FilePathsBase: mkpath, dirname


# ---------------------------------------------------------------------------
# Credentials helper
# ---------------------------------------------------------------------------

"""
    _read_credentials(section) -> (username, password)

Try to read credentials from `credentials.toml` (repo root or
`~/.config/smpl/credentials.toml`). Returns `("", "")` if the file is
absent, the section is missing, or either field is blank — in which case
the caller should fall back to an interactive prompt.
"""
function _read_credentials(section::String) :: Tuple{String,String}
    search_paths = (
        joinpath(dirname(dirname(@__FILE__)), "credentials.toml"),   # repo root
        joinpath(homedir(), ".config", "smpl", "credentials.toml"), # user-level
    )
    for path in search_paths
        isfile(path) || continue
        cfg = TOML.parsefile(path)
        haskey(cfg, section) || continue
        s = cfg[section]
        u = get(s, "username", "")
        p = get(s, "password", "")
        (isempty(u) || isempty(p)) && continue
        return (u, p)
    end
    return ("", "")
end


# ---------------------------------------------------------------------------
# Internal HTTP download helpers
# ---------------------------------------------------------------------------

# Post login credentials and stream the response to a local file.
# Uses HTTP.jl directly (no external wget dependency).
# Validates that the result is an NPZ/ZIP archive (magic bytes PK\x03\x04);
# if the server returns an HTML error/login page the file is removed and an
# informative error is thrown so the user knows to fix their credentials.
function _download_file(url::String, post_data::String, output_file::String)
    output_dir = dirname(output_file)
    mkpath(output_dir)
    open(output_file, "w") do io
        HTTP.request(
            "POST", url,
            ["Content-Type" => "application/x-www-form-urlencoded"],
            post_data;
            response_stream          = io,
            redirect                 = true,
            require_ssl_verification = false,
        )
    end
    # NPZ files are ZIP archives — magic bytes are PK\x03\x04 (0x50 0x4b 0x03 0x04).
    # If the server returned an HTML page (auth failure, wrong URL, etc.) the first
    # bytes will be '<' or similar.  Catch this early with a clear error message.
    magic = open(output_file, "r") do io; read(io, 4); end
    if magic != UInt8[0x50, 0x4b, 0x03, 0x04]
        rm(output_file; force=true)
        error(
            "Downloaded file is not a valid NPZ archive — the server likely returned " *
            "an HTML authentication error page.\n\n" *
            "To fix:\n" *
            "  1. Verify your credentials in credentials.toml (copy from credentials.toml.example).\n" *
            "  2. Delete the bad cache directory so DataDeps re-downloads:\n" *
            "       rm -rf ~/.julia/scratchspaces/124859b0-ceae-595e-8997-d05f6a7a8dfe/datadeps/\n" *
            "  3. Re-run create_smpl_*/create_smplx_*/create_supr_*().\n\n" *
            "URL attempted: $url"
        )
    end
end

# Validate that a file is an NPZ/ZIP archive (magic bytes PK\x03\x04).
# Called from create_smpl/smplx/supr before NPZ.npzread so that stale bad
# files in the DataDeps cache produce an actionable error rather than an
# opaque "not a NPY or NPZ/Zip file" crash.
function _assert_valid_npz(path::String)
    isfile(path) || error("Model file not found: $path")
    magic = open(path, "r") do io; read(io, 4); end
    if magic != UInt8[0x50, 0x4b, 0x03, 0x04]
        rm(path; force=true)
        error(
            "Cached model file is not a valid NPZ archive and has been deleted: $path\n\n" *
            "This usually means the file was downloaded with wrong credentials " *
            "(the server returned an HTML login page instead of the model).\n\n" *
            "To fix:\n" *
            "  1. Check your credentials in credentials.toml.\n" *
            "  2. Re-run create_smpl_*/create_smplx_*/create_supr_*() — " *
            "DataDeps will re-download the missing file."
        )
    end
end

# Prompt for credentials interactively (fallback when credentials.toml is absent).
function _prompt_credentials(model_name::String) :: Tuple{String,String}
    print("$model_name username: ");  username = readline()
    print("$model_name password: ");  password = readline()
    return (username, password)
end

# Load credentials from file or prompt, then URL-encode.
function _get_post_data(section::String, model_name::String) :: String
    (u, p) = _read_credentials(section)
    if isempty(u) || isempty(p)
        (u, p) = _prompt_credentials(model_name)
    end
    return "username=$(escape(u))&password=$(escape(p))"
end

# Download all SMPL model variants (female/male/neutral).
function _fetch_smpl_models(_remote_filepath, local_directorypath)
    post = _get_post_data("smpl", "SMPL")
    base = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=julia"
    for variant in ("SMPL_FEMALE", "SMPL_MALE", "SMPL_NEUTRAL")
        _download_file("$base/$variant.npz", post, joinpath(local_directorypath, "$variant.npz"))
    end
    return local_directorypath
end

# Download all SMPLX model variants.
function _fetch_smplx_models(_remote_filepath, local_directorypath)
    post = _get_post_data("smplx", "SMPLX")
    base = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=julia"
    for variant in ("SMPLX_FEMALE", "SMPLX_MALE", "SMPLX_NEUTRAL")
        _download_file("$base/$variant.npz", post, joinpath(local_directorypath, "$variant.npz"))
    end
    return local_directorypath
end

# Download all SUPR model variants.
function _fetch_supr_models(_remote_filepath, local_directorypath)
    post = _get_post_data("supr", "SUPR")
    base = "https://download.is.tue.mpg.de/download.php?domain=supr&resume=1&sfile"
    paths = (
        ("male/body/SUPR_male.npz",       "SUPR_MALE.npz"),
        ("female/body/SUPR_female.npz",   "SUPR_FEMALE.npz"),
        ("generic/body/SUPR_neutral.npz", "SUPR_NEUTRAL.npz"),
    )
    for (remote, local_name) in paths
        _download_file("$base=$remote", post, joinpath(local_directorypath, local_name))
    end
    return local_directorypath
end


# ---------------------------------------------------------------------------
# DataDeps __init__ — register model download providers
# ---------------------------------------------------------------------------

"""
    __init__()

Registers DataDeps entries for SMPL_models, SMPLX_models, and SUPR_models.
Called automatically when the SMPL module is loaded. The first call to
`create_smpl_*` / `create_smplx_*` / `create_supr_*` will trigger the
credential lookup (credentials.toml → interactive fallback) and download
if the files are not yet cached.
"""
function __init__()
    register(DataDep(
        "SMPL_models",
        "SMPL model files — register at https://smpl.is.tue.mpg.de",
        "https://smpl.is.tue.mpg.de",
        Any;
        fetch_method      = _fetch_smpl_models,
        post_fetch_method = identity,
    ))

    register(DataDep(
        "SMPLX_models",
        "SMPLX model files — register at https://smpl-x.is.tue.mpg.de",
        "https://smpl-x.is.tue.mpg.de",
        Any;
        fetch_method      = _fetch_smplx_models,
        post_fetch_method = identity,
    ))

    register(DataDep(
        "SUPR_models",
        "SUPR model files — register at https://supr.is.tue.mpg.de",
        "https://supr.is.tue.mpg.de",
        Any;
        fetch_method      = _fetch_supr_models,
        post_fetch_method = identity,
    ))
end


# ---------------------------------------------------------------------------
# create_smpl — load SMPL body model from .npz
# ---------------------------------------------------------------------------

"""
    create_smpl(model_path::String) -> BodyModel{Float32, Matrix{Float32}}

Load an SMPL model from a `.npz` file and return a CPU `BodyModel`.

Array layout conversions performed at load time (once):
  - `shapedirs`: (6890, 3, N_b) → reshaped to (6890*3, N_b)  [S matrix]
  - `posedirs`:  (6890*3, N_b)  → transposed to ((N_j-1)*9, 6890*3)  [P matrix]
  - `parents`:   0-indexed UInt32 → 1-indexed Int32
  - `f`:         0-indexed UInt32 → 1-indexed UInt32

The returned model is ready for `smpl_lbs`. Move to GPU with
`Adapt.adapt(CuArray, model)` when CUDA.jl is loaded.
"""
function create_smpl(model_path::String) :: BodyModel{Float32, Matrix{Float32}}
    _assert_valid_npz(model_path)
    d = NPZ.npzread(model_path)

    # shapedirs in NPZ: (6890, 3, N_b) — reshape to (N_v*3, N_b) for S·β matmul
    shapedirs  = Float32.(reshape(d["shapedirs"], 6890*3, :))      # (N_v*3, N_b)
    # posedirs in NPZ: (N_v*3, (N_j-1)*9) — transpose to ((N_j-1)*9, N_v*3) for ψ·P matmul
    posedirs   = Float32.(reshape(d["posedirs"],  6890*3, :)')     # ((N_j-1)*9, N_v*3)

    # parents: 0-indexed in NPZ → 1-indexed Int32 for Julia.
    # The root joint's parent is stored as 0xffffffff (sentinel) in the SMPL NPZ;
    # force it to 0 before conversion so root[1]=1 (self-referencing root convention).
    d["kintree_table"][1, 1] = 0
    parents    = Int32.(d["kintree_table"][1, :]) .+ Int32(1)      # (N_j,)

    # faces: 0-indexed in Python/NPZ → 1-indexed for Julia mesh operations
    faces      = UInt32.(d["f"]) .+ UInt32(1)                     # (N_f, 3)

    return BodyModel{Float32, Matrix{Float32}}(
        Float32.(d["v_template"]),   # (6890, 3)
        shapedirs,                   # (6890*3, N_b)
        posedirs,                    # (207, 6890*3)  since (N_j-1)*9 = 23*9 = 207
        Float32.(d["J_regressor"]),  # (24, 6890)
        Float32.(d["weights"]),      # (6890, 24)
        parents,                     # (24,)
        faces,                       # (N_f, 3)
    )
end

# Convenience wrappers: trigger DataDeps download on first call.
"""
    create_smpl_female() -> BodyModel

Load the female SMPL model, downloading from smpl.is.tue.mpg.de if not cached.
"""
create_smpl_female()  = create_smpl(joinpath(datadep"SMPL_models", "SMPL_FEMALE.npz"))

"""
    create_smpl_male() -> BodyModel

Load the male SMPL model, downloading from smpl.is.tue.mpg.de if not cached.
"""
create_smpl_male()    = create_smpl(joinpath(datadep"SMPL_models", "SMPL_MALE.npz"))

"""
    create_smpl_neutral() -> BodyModel

Load the gender-neutral SMPL model, downloading from smpl.is.tue.mpg.de if not cached.
"""
create_smpl_neutral() = create_smpl(joinpath(datadep"SMPL_models", "SMPL_NEUTRAL.npz"))


# ---------------------------------------------------------------------------
# create_smplx — load SMPLX body model from .npz
# ---------------------------------------------------------------------------

"""
    create_smplx(model_path::String) -> BodyModel{Float32, Matrix{Float32}}

Load an SMPLX model from a `.npz` file and return a CPU `BodyModel`.

SMPLX is structurally identical to SMPL but has 55 joints (vs 24) and
10475 vertices (vs 6890), adding detailed hand and face articulation.
The same `smpl_lbs` function handles both — sizes are inferred from arrays.

Array layout conversions:
  - `kintree_table[1,1]` forced to 0 (SMPLX NPZ quirk: root parent is stored
    as a large sentinel; override to the standard self-parent convention)
  - Same reshape/transpose pattern as `create_smpl`
"""
function create_smplx(model_path::String) :: BodyModel{Float32, Matrix{Float32}}
    _assert_valid_npz(model_path)
    d = NPZ.npzread(model_path)

    # SMPLX NPZ quirk: root joint's parent entry is a large out-of-range value;
    # override to 0 (0-indexed) so +1 converts it to 1 (1-indexed root convention).
    d["kintree_table"][1, 1] = 0

    shapedirs  = Float32.(reshape(d["shapedirs"], 10475*3, :))     # (N_v*3, N_b)
    posedirs   = Float32.(reshape(d["posedirs"],  10475*3, :)')    # ((N_j-1)*9, N_v*3)
    parents    = Int32.(d["kintree_table"][1, :]) .+ Int32(1)      # (55,)
    faces      = UInt32.(d["f"]) .+ UInt32(1)                     # (N_f, 3)

    return BodyModel{Float32, Matrix{Float32}}(
        Float32.(d["v_template"]),   # (10475, 3)
        shapedirs,                   # (10475*3, N_b)
        posedirs,                    # (486, 10475*3)  since (55-1)*9 = 486
        Float32.(d["J_regressor"]),  # (55, 10475)
        Float32.(d["weights"]),      # (10475, 55)
        parents,                     # (55,)
        faces,                       # (N_f, 3)
    )
end

"""
    create_smplx_female() -> BodyModel

Load the female SMPLX model, downloading from smpl-x.is.tue.mpg.de if not cached.
"""
create_smplx_female()  = create_smplx(joinpath(datadep"SMPLX_models", "SMPLX_FEMALE.npz"))

"""
    create_smplx_male() -> BodyModel

Load the male SMPLX model, downloading from smpl-x.is.tue.mpg.de if not cached.
"""
create_smplx_male()    = create_smplx(joinpath(datadep"SMPLX_models", "SMPLX_MALE.npz"))

"""
    create_smplx_neutral() -> BodyModel

Load the gender-neutral SMPLX model, downloading from smpl-x.is.tue.mpg.de if not cached.
"""
create_smplx_neutral() = create_smplx(joinpath(datadep"SMPLX_models", "SMPLX_NEUTRAL.npz"))


# ---------------------------------------------------------------------------
# create_supr — load SUPR body model from .npz
# ---------------------------------------------------------------------------

"""
    create_supr(model_path::String) -> SUPRModel{Float32, Matrix{Float32}}

Load a SUPR model from a `.npz` file and return a CPU `SUPRModel`.

SUPR (Sparse Unified Part-Based Human Body Representation) differs from SMPL:
  - 75 joints (vs 24/55), 10475 vertices
  - Pose features: 4D quaternion per joint (N_j*4 total) instead of 9D rot-mat
  - Joint regressor: general affine map (3*N_j, N_v*3) + bias, not scalar weights

SUPR-specific preprocessing at load time:
  - `shapedirs`: (N_v, 3, N_b) → permutedims + reshape → (N_v*3, N_b)
  - `posedirs`:  (N_v, 3, N_p) → permutedims + reshape → (N_p, N_v*3)
  - `J_regressor`: sparse (3*N_j, N_v*3+1) → dense, split into body + bias:
      J_body = Matrix(J_reg[:, 1:end-1])  shape (3*N_j, N_v*3)
      J_bias = reshape(J_reg[:, end], 3, N_j)'  shape (N_j, 3)
  - `parents`: 0-indexed kintree_table → 1-indexed Int32, all N_j entries
"""
function create_supr(model_path::String) :: SUPRModel{Float32, Matrix{Float32}}
    _assert_valid_npz(model_path)
    d = NPZ.npzread(model_path)

    N_v    = size(d["v_template"], 1)   # 10475
    N_j    = 75                          # SUPR always has 75 joints
    N_b    = size(d["shapedirs"], 3)    # number of shape components

    # shapedirs in SUPR NPZ: (N_v, 3, N_b) — different from SMPL's (N_v, 3, N_b)
    # but with a different memory layout. Permute dims to group vertex coordinates,
    # then reshape to flat (N_v*3, N_b) matching the BodyModel S·β convention.
    shapedirs = Float32.(reshape(permutedims(d["shapedirs"], (2, 1, 3)), N_v*3, N_b))

    # posedirs in SUPR NPZ: (N_v, 3, N_p) where N_p = N_j * 4 (quaternion features)
    # Permute + reshape → (N_v*3, N_p), then transpose → (N_p, N_v*3) for ψ·P matmul
    N_p       = size(d["posedirs"], 3)  # = N_j * 4
    posedirs  = Float32.(reshape(permutedims(d["posedirs"], (2, 1, 3)), N_v*3, N_p)')

    # J_regressor: SUPR stores as sparse (3*N_j, N_v*3+1).
    # The last column encodes an additive joint position bias b.
    # Operation: J = reshape(J_reg * [v_flat; 1], 3, N_j)'
    #           = reshape(J_body * v_flat + j_bias_flat, 3, N_j)'
    J_reg_full = Matrix(Float32.(d["J_regressor"]))    # dense (3*N_j, N_v*3+1)
    J_body     = J_reg_full[:, 1:end-1]                # (3*N_j, N_v*3) — body weights
    J_bias     = reshape(J_reg_full[:, end], 3, N_j)'  # (N_j, 3)       — additive bias

    # parents: SUPR kintree_table has N_j entries (0-indexed).
    # Force root sentinel to 0 before conversion (same quirk as SMPL/SMPLX).
    d["kintree_table"][1, 1] = 0
    parents = Int32.(d["kintree_table"][1, :]) .+ Int32(1)   # (N_j,)

    faces   = UInt32.(d["f"]) .+ UInt32(1)                   # (N_f, 3)

    return SUPRModel{Float32, Matrix{Float32}}(
        Float32.(d["v_template"]),  # (N_v, 3)
        shapedirs,                  # (N_v*3, N_b)
        posedirs,                   # (N_p, N_v*3)
        Float32.(J_body),           # (3*N_j, N_v*3)
        Float32.(J_bias),           # (N_j, 3)
        Float32.(d["weights"]),     # (N_v, N_j)
        parents,                    # (N_j,)
        faces,                      # (N_f, 3)
    )
end

"""
    create_supr_female() -> SUPRModel

Load the female SUPR model, downloading from supr.is.tue.mpg.de if not cached.
"""
create_supr_female()  = create_supr(joinpath(datadep"SUPR_models", "SUPR_FEMALE.npz"))

"""
    create_supr_male() -> SUPRModel

Load the male SUPR model, downloading from supr.is.tue.mpg.de if not cached.
"""
create_supr_male()    = create_supr(joinpath(datadep"SUPR_models", "SUPR_MALE.npz"))

"""
    create_supr_neutral() -> SUPRModel

Load the gender-neutral SUPR model, downloading from supr.is.tue.mpg.de if not cached.
"""
create_supr_neutral() = create_supr(joinpath(datadep"SUPR_models", "SUPR_NEUTRAL.npz"))
