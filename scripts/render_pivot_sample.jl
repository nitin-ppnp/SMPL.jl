# scripts/render_pivot_sample.jl
#
# Demo: render pivot joint probability visualization for an ACCAD walking sequence.
#
# Loads the same motion from both AMASS dataset variants:
#   - AMASS_SMPLX_NEUTRAL_Yup_smplFormat  (Y-up world coords, requires up=:y)
#   - AMASS_SMPLX_NEUTRAL_smplFormat      (Z-up world coords, default up=:z)
#
# Both should produce visually identical output when the up-axis is handled correctly.
#
# Output files (written to scripts/):
#   pivot_frame_max_yup.png        — :max mode, Yup dataset  (red sphere)
#   pivot_frame_max_zup.png        — :max mode, Zup dataset  (should look identical)
#   pivot_frame_threshold_yup.png  — :threshold mode, Yup dataset
#   pivot_max_mode.mp4             — :max mode video (first 5 s, Yup dataset)
#   pivot_threshold_mode.mp4       — :threshold mode video (first 5 s, Yup dataset)
#
# Usage:
#   julia scripts/render_pivot_sample.jl
#
# Requires CairoMakie (headless; no display needed).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SMPL, CairoMakie

const MOTION_ROOT_YUP = "//ds1823pl/storage/datasets/AMASS/AMASS_SMPLX_NEUTRAL_Yup_smplFormat"
const MOTION_ROOT_ZUP = "//ds1823pl/storage/datasets/AMASS/AMASS_SMPLX_NEUTRAL_smplFormat"
const PIVOT_ROOT      = "//ds1823pl/storage/datasets/AMASS/AMASS_SMPLX_NEUTRAL_YUP_pivotLabels_V1"
const SUBJECT         = joinpath("ACCAD", "Female1Walking_c3d")
const STEM            = "B10_-_walk_turn_left_(45)_stageii"

println("Loading SMPLX neutral model...")
model = create_smplx_neutral()

# ---------------------------------------------------------------------------
# Load Yup dataset — explicit up=:y (root rotation does not encode Y→Z flip)
# ---------------------------------------------------------------------------
println("Loading Yup motion sequence + pivot labels...")
seq_yup = load_motion(joinpath(MOTION_ROOT_YUP, SUBJECT, STEM * ".smpl");
                      pivot_labels_path = joinpath(PIVOT_ROOT, SUBJECT, STEM * ".npy"),
                      up                = :y)
N = size(seq_yup.poses, 1)
println("  Frames: $N  |  up=$(seq_yup.up)  |  Pivot: $(size(seq_yup.pivot_joints))")

# ---------------------------------------------------------------------------
# Load Zup dataset — default up=:z
# ---------------------------------------------------------------------------
println("Loading Zup motion sequence...")
seq_zup = load_motion(joinpath(MOTION_ROOT_ZUP, SUBJECT, STEM * ".smpl"))
println("  Frames: $(size(seq_zup.poses,1))  |  up=$(seq_zup.up)")

# ---------------------------------------------------------------------------
# Single-frame renders — mid-sequence frame
# ---------------------------------------------------------------------------
mid = N ÷ 2

println("Rendering pivot_frame_max_yup.png  (frame $mid, Yup, :max)...")
render_frame(model, seq_yup, mid,
             joinpath(@__DIR__, "pivot_frame_max_yup.png");
             resolution = (960, 720), pivot_mode = :max)

println("Rendering pivot_frame_max_zup.png  (frame $mid, Zup, no pivot)...")
render_frame(model, seq_zup, mid,
             joinpath(@__DIR__, "pivot_frame_max_zup.png");
             resolution = (960, 720))

println("Rendering pivot_frame_threshold_yup.png  (:threshold, threshold=0.5)...")
render_frame(model, seq_yup, mid,
             joinpath(@__DIR__, "pivot_frame_threshold_yup.png");
             resolution      = (960, 720),
             pivot_mode      = :threshold,
             pivot_threshold = 0.5f0)

# ---------------------------------------------------------------------------
# Video renders — first 5 s of Yup sequence
# ---------------------------------------------------------------------------
n5 = min(round(Int, 5 * seq_yup.fps), N)
short = MotionSequence(
    seq_yup.poses[1:n5, :],
    seq_yup.betas,
    seq_yup.trans[1:n5, :],
    seq_yup.fps,
    seq_yup.model_type,
    seq_yup.gender,
    seq_yup.up,
    seq_yup.pivot_joints[1:n5, :],
)

println("Recording pivot_max_mode.mp4  ($n5 frames, :max mode)...")
record_motion(model, short,
              joinpath(@__DIR__, "pivot_max_mode.mp4");
              resolution = (960, 720), pivot_mode = :max)

println("Recording pivot_threshold_mode.mp4  (:threshold, threshold=0.5)...")
record_motion(model, short,
              joinpath(@__DIR__, "pivot_threshold_mode.mp4");
              resolution      = (960, 720),
              pivot_mode      = :threshold,
              pivot_threshold = 0.5f0)

println("Done. Output files written to: ", @__DIR__)
