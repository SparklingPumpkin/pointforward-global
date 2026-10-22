"""Optional all-visible-view refresh of one persistent Gaussian set per actor.

The initial 12 views are unchanged. Later 12-view groups cover every remaining
visible frame-camera pair and requery existing centres through the original
projection/fusion/decoder. No independently reconstructed cars are appended.
"""
from __future__ import annotations

from dataclasses import replace
import torch

from scripts.deliver_continuous_scene import (
    blend_dynamic_gaussians, build_queries_from_dynamic_gaussians,
    global_dynamic_to_local_tracks, replace_gaussians, select_gaussians,
)


def expand_all_frame_jobs(jobs, candidates, angle_weight):
    from scripts.build_track_once_dynamic import select_diverse_large_view_pass
    expanded = []
    for job in jobs:
        pool = list(candidates[job['track_id']])
        if job['allowed_frames'] is not None:
            pool = [v for v in pool if v['frame'] in job['allowed_frames']]
        first = select_diverse_large_view_pass(pool, 12, angle_weight, 0)
        used = {(v['frame'], v['camera']) for v in first}
        remaining = sorted(
            (v for v in pool if (v['frame'], v['camera']) not in used),
            key=lambda v: (-v['frame'], v['camera']),
        )
        groups = [first] + [remaining[k:k+12] for k in range(0, len(remaining), 12)]
        for index, views in enumerate(groups):
            original_count = len(views)
            views = list(views)
            while len(views) < 12:
                views.append(dict(views[(len(views)-original_count) % original_count]))
            expanded.append(dict(job, selected_views=views, view_group_index=index,
                                 unique_group_view_count=original_count,
                                 view_group_count=len(groups)))
    return expanded


def persistent_queries(state, scene, depth, track_id, global_slot, global_boxes):
    local, _ = global_dynamic_to_local_tracks(
        state, scene.boxes, [track_id], global_boxes, {track_id: global_slot},
    )
    return build_queries_from_dynamic_gaussians(local, scene, depth, depth_threshold=1.0)


def integrate_refresh(state, weights, update, source_index):
    evidence = update.extras['best_observation_distance'].clamp_min(1.0).square().reciprocal()
    if state is None:
        return update, evidence
    if source_index is None or len(source_index) != update.count:
        raise ValueError('A refresh must retain a one-to-one persistent point association')
    old = select_gaussians(state, source_index)
    total = weights[source_index] + evidence
    alpha = evidence / total.clamp_min(1e-12)
    refreshed = blend_dynamic_gaussians(old, update, alpha)
    # Dynamic actors are not subject to the static frozen-position restriction.
    refreshed = replace(refreshed, means=torch.lerp(old.means, update.means, alpha[:, None]))
    weights[source_index] = total
    return replace_gaussians(state, source_index, refreshed), weights
