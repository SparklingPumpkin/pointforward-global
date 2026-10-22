"""Pretrained human head, with source FFR backbone and joint runtime skinning."""
from collections import OrderedDict
from pathlib import Path
from types import MethodType

import torch
from pedestrian_reference.models.smpl import SMPL, NUM_VERTICES
from pedestrian_reference.models.pointforward import PointForward as HumanSourceModel
from .gaussian_head import GaussianDecoder
from .rotation import matrix_to_quaternion, quaternion_multiply


def attach_pedestrians(model, cfg, device):
    options = cfg.get("pedestrians", {})
    if not options.get("enabled", False):
        return model
    model.smpl = SMPL(options["smpl_model_path"]).to(device)
    model._decode_humans = MethodType(HumanSourceModel._decode_humans, model)
    model.human_decoder = GaussianDecoder(
        in_dim=model.fusion.out_dim, color_dim=cfg["model"].get("color_dim", 128),
        hidden=cfg["model"].get("hidden", 128) * 2, max_offset=0.05,
        scale_init=0.01, max_scale=0.1, grad_checkpoint=False,
    ).to(device)
    state = torch.load(options["checkpoint"], map_location="cpu", weights_only=False)["model"]
    head = {k.removeprefix("human_decoder."): v for k, v in state.items() if k.startswith("human_decoder.")}
    model.human_decoder.load_state_dict(head, strict=True)
    model.human_decoder.requires_grad_(False)
    model.pedestrian_weight_keys = len(head)
    model.pedestrian_runtime = None
    path = Path(options["tracks_file"])
    if path.exists():
        raw = torch.load(path, map_location=device, weights_only=True)
        if "slot_offset" in raw:
            model.pedestrian_runtime = raw
    model.pedestrian_skin_cache = OrderedDict()
    print(f"SMPL enabled: loaded {len(head)} human head tensors; original backbone untouched", flush=True)
    return model


def skin_pedestrians(model, gaussians, posed, frame):
    raw = getattr(model, "pedestrian_runtime", None)
    if raw is None:
        return posed
    offset = int(raw["slot_offset"])
    mask = gaussians.instance_id >= offset
    if not bool(mask.any()):
        return posed
    slot = min(int(frame), raw["pose"].shape[1] - 1)
    cache = model.pedestrian_skin_cache
    if slot not in cache:
        with torch.no_grad():
            _, rots, trans = model.smpl.pose_to_world(
                raw["betas"], raw["pose"][None, :, slot], raw["centers"][None, :, slot])
            cache[slot] = (rots[0].flatten(0, 1), trans[0].flatten(0, 1))
        if len(cache) > 8:
            cache.popitem(last=False)
    rots, trans = cache[slot]
    expected_ids = torch.arange(len(raw["betas"]), device=rots.device).repeat_interleave(NUM_VERTICES) + offset
    if not torch.equal(gaussians.instance_id[mask], expected_ids):
        raise ValueError("Human vertex order/identity changed; refusing incorrect skinning")
    out = dict(posed)
    out["means"] = posed["means"].clone()
    out["quats"] = posed["quats"].clone()
    out["opacities"] = posed["opacities"].clone()
    out["means"][mask] = (rots @ gaussians.means[mask].float().unsqueeze(-1)).squeeze(-1) + trans
    out["quats"][mask] = quaternion_multiply(matrix_to_quaternion(rots), gaussians.quats[mask].float())
    # Human visibility is pose validity; do not apply rigid Gaussian time decay.
    present = raw["pose_valid"][:, slot].repeat_interleave(NUM_VERTICES)
    out["opacities"][mask] = gaussians.opacities[mask] * present.to(gaussians.opacities.dtype)
    return out
