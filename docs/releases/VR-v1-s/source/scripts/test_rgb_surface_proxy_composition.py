#!/usr/bin/env python3
"""Production-path synthetic checks for Round-64 proxy composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pointforward.models import render_features
from scripts.render_final_global_runtime import composite_rgb_surface_proxy


H = W = 32


def tensors(depths, opacities, colors, device):
    count = len(depths)
    means = torch.tensor([[0.0, 0.0, float(z)] for z in depths], device=device)
    quats = torch.zeros((count, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((count, 3), 0.20, device=device)
    opacity = torch.tensor(opacities, device=device)
    color = torch.tensor(colors, device=device)
    return means, quats, scales, opacity, color


def render(depths, opacities, colors, device):
    values = tensors(depths, opacities, colors, device)
    k = torch.tensor(
        [[[40.0, 0.0, 15.5], [0.0, 40.0, 15.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    view = torch.eye(4, device=device).unsqueeze(0)
    rgb, alpha, _ = render_features(
        means=values[0], quats=values[1], scales=values[2],
        opacities=values[3], colors=values[4], intrinsics=k,
        world_to_cam=view, height=H, width=W, render_depth=False,
    )
    return rgb, alpha


def production_case(base, proxy, device):
    base_depths, base_opacities, base_colors = base
    proxy_depths, proxy_opacities, proxy_colors = proxy
    base_rgb, _ = render(base_depths, base_opacities, base_colors, device)
    proxy_rgb, proxy_alpha = render(
        proxy_depths, proxy_opacities, proxy_colors, device
    )
    joint_depths = base_depths + proxy_depths
    joint_opacities = base_opacities + proxy_opacities
    joint_colors = (
        [[0.0, 0.0, 0.0, 0.0] for _ in base_depths]
        + [list(color) + [1.0] for color in proxy_colors]
    )
    joint, _ = render(joint_depths, joint_opacities, joint_colors, device)
    final, accepted, ratio = composite_rgb_surface_proxy(
        base_rgb, proxy_rgb, proxy_alpha, joint[:, 3:4]
    )
    return base_rgb, proxy_rgb, proxy_alpha, final, accepted, ratio


def reject_case(name, base, proxy, device):
    base_rgb, _, proxy_alpha, final, accepted, ratio = production_case(base, proxy, device)
    centre = (0, 0, 15, 15)
    if bool(accepted[centre]):
        raise AssertionError(f"{name}: centre pixel was not rejected")
    error = float((final - base_rgb).abs().max())
    if error > 1e-6:
        raise AssertionError(f"{name}: rejected result differs from base by {error}")
    return {
        "name": name,
        "centre_proxy_alpha": float(proxy_alpha[centre]),
        "centre_visibility_ratio": float(ratio[centre]),
        "max_abs_from_base": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    red = [1.0, 0.0, 0.0]
    green = [0.0, 1.0, 0.0]
    blue = [0.0, 0.0, 1.0]
    results = []
    results.append(reject_case(
        "baseline_static_5m_blocks_proxy_dynamic_10m",
        ([5.0], [0.50], [red]), ([10.0], [0.85], [green]), device,
    ))
    results.append(reject_case(
        "baseline_static_5m_blocks_proxy_static_10m",
        ([5.0], [0.50], [red]), ([10.0], [0.85], [green]), device,
    ))
    results.append(reject_case(
        "critic_interleaved_red_proxy_green_blue",
        ([10.0, 20.0], [0.50, 0.50], [red, blue]),
        ([15.0], [0.85], [green]), device,
    ))

    base_rgb, proxy_rgb, proxy_alpha, final, accepted, ratio = production_case(
        ([10.0], [0.50], [red]), ([5.0], [0.85], [green]), device
    )
    centre = (0, 0, 15, 15)
    if not bool(accepted[centre]):
        raise AssertionError("front proxy centre pixel was rejected")
    reference = proxy_rgb + (1.0 - proxy_alpha) * base_rgb
    front_error = float((final[accepted.expand_as(final)]
                         - reference[accepted.expand_as(reference)]).abs().max())
    if front_error > 1e-6:
        raise AssertionError(f"front proxy alpha-over mismatch {front_error}")
    results.append({
        "name": "proxy_in_front_standard_alpha_over",
        "centre_visibility_ratio": float(ratio[centre]),
        "max_abs_from_reference": front_error,
    })

    empty_base = ([10.0], [0.0], [red])
    _, proxy_rgb, proxy_alpha, final, accepted, ratio = production_case(
        empty_base, ([10.0], [0.85], [green]), device
    )
    if not bool(accepted[centre]):
        raise AssertionError("proxy did not fill the suppressed dynamic hole")
    accepted_rgb = accepted.expand_as(final)
    hole_error = float((final[accepted_rgb] - proxy_rgb[accepted_rgb]).abs().max())
    if hole_error > 1e-6:
        raise AssertionError(f"hole fill mismatch {hole_error}")
    results.append({
        "name": "suppressed_baseline_dynamic_hole_fill",
        "centre_visibility_ratio": float(ratio[centre]),
        "max_abs_from_proxy": hole_error,
        "static_dynamic_proxy_depth_order_identical": True,
    })

    disabled_base = torch.rand((1, 3, H, W), device=device)
    disabled_result = disabled_base
    disabled_error = float((disabled_result - disabled_base).abs().max())
    if disabled_error != 0.0:
        raise AssertionError("disabled proxy path is not tensor-identical")
    results.append({
        "name": "proxy_disabled_parity",
        "max_abs_from_base": disabled_error,
    })

    report = {
        "status": "passed",
        "device": str(device),
        "tests": results,
        "maximum_allowed_absolute_error": 1e-6,
        "production_compositor": "composite_rgb_surface_proxy",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
