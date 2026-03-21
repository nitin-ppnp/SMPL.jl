# scripts/test_viz.jl
# Test visualization functions against real AMASS data.
# Renders frames to PNG and a short video clip using CairoMakie (headless).
#
# Usage:
#   julia scripts/test_viz.jl
#
# For the interactive viewer, run in a Julia REPL:
#   using SMPL, GLMakie
#   model = create_smplx_neutral()
#   seq = load_motion("//ds1823pl/storage/datasets/AMASS/AMASS_SMPLX_NEUTRAL_smplFormat/ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_(45)_stageii.smpl")
#   viz_motion(model, seq)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SMPL, CairoMakie

const AMASS_ROOT = "//ds1823pl/storage/datasets/AMASS/AMASS_SMPLX_NEUTRAL_smplFormat"
const SEQ1 = joinpath(AMASS_ROOT, "ACCAD", "Female1Walking_c3d",
                      "B10_-_walk_turn_left_(45)_stageii.smpl")
const SEQ2 = joinpath(AMASS_ROOT, "ACCAD", "Female1Walking_c3d",
                      "B12_-_walk_turn_right_(90)_stageii.smpl")

println("=" ^ 60)
println("Loading SMPLX neutral model...")
model = create_smplx_neutral()
println("  OK — $(size(model.v_template, 1)) vertices, $(size(model.J_regressor, 1)) joints")

println()
println("Loading motion sequences...")
seq1 = load_motion(SEQ1)
seq2 = load_motion(SEQ2)

println("  seq1: $(size(seq1.poses, 1)) frames @ $(seq1.fps) fps | $(seq1.model_type) | $(seq1.gender)")
println("  seq2: $(size(seq2.poses, 1)) frames @ $(seq2.fps) fps | $(seq2.model_type) | $(seq2.gender)")

outdir = @__DIR__

# -----------------------------------------------------------------------
# Single-frame renders (CairoMakie, headless)
# -----------------------------------------------------------------------
println()
println("Rendering frames with CairoMakie...")

out1 = joinpath(outdir, "frame_001.png")
render_frame(model, seq1, 1, out1; resolution=(960, 720))
println("  -> frame_001.png  (frame 1)")

mid = size(seq1.poses, 1) ÷ 2
out2 = joinpath(outdir, "frame_mid.png")
render_frame(model, seq1, mid, out2; resolution=(960, 720))
println("  -> frame_mid.png  (frame $mid / $(size(seq1.poses,1)))")

# Last frame
last_frame = size(seq1.poses, 1)
out3 = joinpath(outdir, "frame_last.png")
render_frame(model, seq1, last_frame, out3; resolution=(960, 720))
println("  -> frame_last.png  (frame $last_frame)")

# -----------------------------------------------------------------------
# Side-by-side render: seq1 vs seq2 at their midpoints
# -----------------------------------------------------------------------
println()
println("Rendering side-by-side comparison (seq1 mid vs seq2 mid)...")
out_sbs = joinpath(outdir, "sidebyside.png")

fig = Figure(size=(1920, 720))
for (col, (seq, label)) in enumerate(((seq1, "seq1"), (seq2, "seq2")))
    ax = LScene(fig[1, col], show_axis=false)
    frame_idx = size(seq.poses, 1) ÷ 2
    out = SMPL.smpl_lbs(model,
                        reshape(seq.betas, :),
                        reshape(seq.poses[frame_idx, :], :),
                        seq.trans[frame_idx, :])
    mesh!(ax, out.vertices, out.faces; color=:lightblue, shading=true)
    Label(fig[2, col], label; tellwidth=false)
end
save(out_sbs, fig)
println("  -> sidebyside.png")

# -----------------------------------------------------------------------
# Short video clip (first 5s)
# -----------------------------------------------------------------------
println()
n5s = min(round(Int, 5 * seq1.fps), size(seq1.poses, 1))
short_seq = MotionSequence(
    seq1.poses[1:n5s, :],
    seq1.betas,
    seq1.trans[1:n5s, :],
    seq1.fps,
    seq1.model_type,
    seq1.gender,
    seq1.up,
)
outvid = joinpath(outdir, "test_motion.mp4")
println("Recording 5s clip ($n5s frames @ $(seq1.fps) fps)...")
record_motion(model, short_seq, outvid; resolution=(960, 720))
println("  -> test_motion.mp4")

println()
println("=" ^ 60)
println("Done. Output files in scripts/:")
println("  frame_001.png, frame_mid.png, frame_last.png")
println("  sidebyside.png")
println("  test_motion.mp4")
println()
println("For interactive visualization, run in a Julia REPL:")
println("  using SMPL, GLMakie")
println("  model = create_smplx_neutral()")
println("  seq = load_motion(\"$SEQ1\")")
println("  viz_motion(model, seq)")
println("=" ^ 60)
