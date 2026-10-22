# VR-v1-s strict cold compact-union result

- Raw preprocessing + FF + PCA + optimizer initialization, 1670 full3cam steps and final checkpoint: **1485.371231 s (24.756 min)**.
- Full156x3 originalbox vehicle PSNR **28.329244 dB**; global **31.131654 dB**.
- Acceptance: **not met**; no rounding of29dB threshold.
- Entire raw-to-video+PLY delivery: 1609.087 s.
- TwoGPUs (0/5); independent static/vehicle FF overlap. PCA regenerated from fresh FF; no fittedscene ancestors.
- Runtime uses32D Gaussianfeatures + adapted/trained U-Net, corrected trajectories and original source compositor. PLY preview is not standalone proof of RGB PSNR.
- Source comparisons at20/80/140 personally inspected; residual nearSUV quality noted in session report.

|Stage|Actual subprocess seconds|
|---|---:|
|geometry_prepare|116.464|
|rgb_masks|51.242|
|geometry_fit_filter|108.557|
|static_build|74.333|
|vehicle_build|170.793|
|pedestrian_build|22.374|
|feature_projection|9.481|
|joint|1006.387|
|render_full|118.768|
|ply_export|4.850|

Static/vehicle rows overlap and must not be summed as walltime. Raw stopwatch alone determines acceptance. A1480s budget request includes final saving in actual endpoint; options and realizedsteps are retained. See input_provenance.md for inherited metadata label caveat.

[GT/predicted video](final_global_runtime_gt_vs_predicted_front3.mp4) · [PLY preview](final_global_gaussians_frame000_preview.ply) · [Raw timing](optimization_timing.json)
