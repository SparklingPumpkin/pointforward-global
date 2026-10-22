"""Export unfiltered PointForward projected depth and decode it into one GS PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scripts.export_infinidepth_scene_gaussians import install_python39_infinidepth_compatibility
from scripts.train import build_dataset


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-depths", required=True)
    parser.add_argument("--source-metrics", required=True)
    parser.add_argument("--depth-png", required=True)
    parser.add_argument("--output-ply", required=True)
    args = parser.parse_args()

    import json

    config = yaml.safe_load(Path(args.config).read_text())
    source = json.loads(Path(args.source_metrics).read_text())
    snapshot = source["source_snapshot"]
    arrays = np.load(args.input_depths)
    depth = arrays["gaussian_depth_m"].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    depth_png = Path(args.depth_png)
    output_ply = Path(args.output_ply)
    depth_png.parent.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    display_high = float(np.percentile(depth[valid], 98))
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    artist = axis.imshow(np.ma.array(depth, mask=~valid), cmap="turbo", vmin=0, vmax=display_high)
    axis.set_title("PointForward Gaussian projected expected depth\nNo alpha filtering; camera-Z metres")
    axis.set_axis_off()
    figure.colorbar(artist, ax=axis, label="Depth (m), clipped at P98", shrink=0.85, extend="max")
    figure.savefig(depth_png, dpi=150)
    plt.close(figure)

    dataset = build_dataset(config, "val")
    scene_index, start = dataset.index[snapshot["clip_index"]]
    metadata = dataset.scenes[scene_index]
    window = metadata["frames"][start:start + dataset.clip_length]
    frame = snapshot["snapshot_frame"]
    world_from = np.linalg.inv(dataset._ego_pose(metadata, window[0]))
    batch = dataset._collect(
        metadata,
        [frame],
        {value: index for index, value in enumerate(window)},
        dataset._normalised_times(window),
        dataset.resolution,
        world_from,
        lidar=False,
    )
    view = dataset.cameras.index(source["camera"])
    device = torch.device("cuda")
    image = batch["images"][view:view + 1].to(device)
    intrinsics = batch["intrinsics"][view:view + 1].to(device)
    supplied_depth = torch.from_numpy(np.where(valid, depth, 0.0))[None, None].to(device)
    supplied_mask = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)

    repo = Path(config["depth"]["repo_root"])
    sys.path.insert(0, str(repo))
    install_python39_infinidepth_compatibility(repo)
    from InfiniDepth.gs import GSPixelAlignPredictor, export_ply
    from InfiniDepth.utils.inference_utils import unpack_gaussians_for_export
    from InfiniDepth.utils.io_utils import depth_to_disparity
    from InfiniDepth.utils.model_utils import build_model

    depth_model = build_model(
        "InfiniDepth_DepthSensor",
        model_path=config["depth"]["checkpoint"],
        encoder=config["depth"].get("encoder", "vitl16"),
    ).to(device).eval()
    disparity = depth_to_disparity(supplied_depth)
    _, dino_tokens, _, _ = depth_model.inference_for_gs(
        image=image,
        intrinsics=intrinsics,
        gt_depth=disparity,
        gt_depth_mask=supplied_mask,
        prompt_depth=disparity,
        prompt_mask=supplied_mask,
        sky_mask=None,
        sample_point_num=1,
        coord_deterministic_sampling=True,
    )

    identity = torch.eye(4, device=device).unsqueeze(0)
    predictor = GSPixelAlignPredictor(dino_feature_dim=dino_tokens.shape[-1]).to(device)
    predictor.load_from_infinidepth_gs_checkpoint(
        str(repo / "checkpoints/gs/infinidepth_depthsensor_gs.ckpt")
    )
    predictor.eval()
    gaussians = predictor(image, supplied_depth, dino_tokens, intrinsics, identity)
    means, harmonics, opacities, scales, rotations = unpack_gaussians_for_export(gaussians)
    height, width = depth.shape
    export_ply(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        path=str(output_ply),
        scales=scales,
        rotations=rotations,
        focal_length_px=(float(intrinsics[0, 0, 0]), float(intrinsics[0, 1, 1])),
        principal_point_px=(float(intrinsics[0, 0, 2]), float(intrinsics[0, 1, 2])),
        image_shape=(height, width),
        extrinsic_matrix=identity[0],
        shift_to_center=False,
    )
    print(depth_png)
    print(output_ply)


if __name__ == "__main__":
    main()