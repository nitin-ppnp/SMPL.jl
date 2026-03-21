# Installation

## Installing the Package

SMPL.jl is not yet registered in the Julia General registry. Install it directly from GitHub:

```julia
] add https://github.com/nitin-ppnp/SMPL.jl
```

Or, for development:

```julia
] dev https://github.com/nitin-ppnp/SMPL.jl
```

**Requirements:** Julia 1.9 or later.

## Downloading Model Files

Body model files are gated by registration — you must create an account at each model's website before downloading. SMPL.jl handles the download automatically on first use via DataDeps.jl.

| Model  | Registration URL |
|--------|-----------------|
| SMPL   | https://smpl.is.tue.mpg.de |
| SMPLX  | https://smpl-x.is.tue.mpg.de |
| SUPR   | https://supr.is.tue.mpg.de |

### Interactive Download (default)

The first call to `create_smpl_*`, `create_smplx_*`, or `create_supr_*` triggers a credential prompt:

```
SMPL username: your_email@example.com
SMPL password: ••••••••
```

Files are cached in `~/.julia/datadeps/` and the prompt never appears again.

### Non-Interactive Download via `credentials.toml`

To skip the interactive prompt (useful in scripts or notebooks), copy the example file and fill in your credentials:

```bash
cp credentials.toml.example credentials.toml
# Edit credentials.toml with your usernames and passwords
```

The file structure:

```toml
[smpl]
username = "your_email@example.com"
password = "your_password"

[smplx]
username = "your_email@example.com"
password = "your_password"

[supr]
username = "your_email@example.com"
password = "your_password"
```

`credentials.toml` is listed in `.gitignore` and will never be accidentally committed.

SMPL.jl also checks `~/.config/smpl/credentials.toml` as a user-level fallback.

## Running Tests

```julia
] test SMPL
```

Tests compare `smpl_lbs` output against reference NumPy arrays with a tolerance of `1e-5`. The SMPL and SMPLX tests run automatically; the SUPR test requires SUPR model files to be downloaded first.

## Optional Dependencies

Install backends to enable visualization and GPU support:

```julia
# Interactive visualization
] add GLMakie

# Headless rendering (servers, CI)
] add CairoMakie

# Browser-based (Jupyter / Pluto notebooks)
] add WGLMakie

# GPU acceleration (NVIDIA)
] add CUDA
```

The relevant extensions (`MakieExt`, `AdaptExt`) load automatically when these packages are in your environment.
```
