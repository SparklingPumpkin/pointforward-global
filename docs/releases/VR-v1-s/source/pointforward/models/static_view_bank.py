"""Optional feedforward static feature refresh from a scene-wide view bank.

The trained fusion still receives four chronological frame groups and the
original camera count. Only frame selection varies per query; scene geometry,
opacity, covariance and dynamic assets are not changed.
"""
from __future__ import annotations

import time
import torch
from torch import Tensor

from .fusion import project_points, _sample_depth, _sample_mask


def sample_selected_views(flat_maps: Tensor, map_shape: tuple[int, int],
                          view_ids: Tensor, uv: Tensor,
                          image_size: tuple[int, int]) -> Tensor:
    """Bilinear lookup with a different source image per query/view slot.

    flat_maps is (V*Hf*Wf,C) in view/row/column order. This uses exactly the
    align_corners=True coordinate convention of backbone.sample_features.
    """
    hf, wf = map_shape
    height, width = image_size
    x = uv[..., 0] * ((wf - 1) / (width - 1))
    y = uv[..., 1] * ((hf - 1) / (height - 1))
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = x - x0, y - y0
    output = flat_maps.new_zeros((*view_ids.shape, flat_maps.shape[-1]))
    for ix, iy, weight in ((x0,y0,(1-dx)*(1-dy)),
                           (x0+1,y0,dx*(1-dy)),
                           (x0,y0+1,(1-dx)*dy),
                           (x0+1,y0+1,dx*dy)):
        valid = (ix >= 0) & (ix < wf) & (iy >= 0) & (iy < hf)
        index = view_ids * (hf*wf) + iy.clamp(0,hf-1)*wf + ix.clamp(0,wf-1)
        output.add_(flat_maps[index] * (weight*valid).unsqueeze(-1))
    return output


@torch.no_grad()
def refresh_static_features(model, gaussians, *, images: Tensor,
                            features: Tensor, depth: Tensor,
                            intrinsics: Tensor, cam_to_world: Tensor,
                            frame_numbers: Tensor, ego_to_world: Tensor,
                            camera_count: int, dynamic_masks: Tensor | None,
                            alpha: float = 1.0, depth_tolerance: float = 1.0,
                            frame_count: int = 4, chunk_size: int = 16384) -> dict:
    """Refresh only static 128D colors using nearest reliable frame groups.

    Views must be sorted by (frame,camera), with one complete camera group per
    frame. Points without any reliable view retain their original features.
    """
    if not 0 <= alpha <= 1 or depth_tolerance <= 0 or chunk_size <= 0:
        raise ValueError("Invalid view-bank parameters")
    if bool((gaussians.instance_id >= 0).any()):
        raise ValueError("Static view bank accepts static Gaussians only")
    v, _, height, width = images.shape
    f = len(frame_numbers)
    if v != f*camera_count or f < frame_count or len(ego_to_world) != f:
        raise ValueError("View bank requires complete sorted frame/camera groups")
    if model.fusion._phi_v_views != frame_count*camera_count:
        raise ValueError("Selected bank width must equal trained phi_v view count")
    world_to_cam = torch.linalg.inv(cam_to_world)
    feature_shape = tuple(features.shape[-2:])
    flat_features = features.permute(0,2,3,1).reshape(-1,features.shape[1]).contiguous()
    flat_rgb = images.permute(0,2,3,1).reshape(-1,3).contiguous()
    origins = cam_to_world[:, :3, 3]
    camera_slots = torch.arange(camera_count,device=images.device).view(1,-1,1)
    start = time.perf_counter()
    updated = 0
    reliable_slots = 0
    for lo in range(0,gaussians.count,chunk_size):
        hi = min(lo+chunk_size,gaussians.count)
        xyz = gaussians.means[lo:hi]
        n = len(xyz)
        points = xyz.unsqueeze(0).expand(v,-1,-1)
        uv, projected_depth, in_front = project_points(points,intrinsics,world_to_cam)
        ref_depth = _sample_depth(depth,uv,height,width)
        difference = (projected_depth-ref_depth).abs()
        valid = (in_front & (uv[...,0]>=0) & (uv[...,0]<=width-1)
                 & (uv[...,1]>=0) & (uv[...,1]<=height-1)
                 & ref_depth.isfinite() & (ref_depth>0) & (difference<=depth_tolerance))
        if dynamic_masks is not None:
            valid &= ~_sample_mask(dynamic_masks,uv,height,width)
        distance = (points-origins[:,None]).square().sum(-1).sqrt()
        group_valid = valid.reshape(f,camera_count,n).any(1)
        group_distance = distance.masked_fill(~valid,float('inf')).reshape(f,camera_count,n).amin(1)
        fallback_distance = distance.reshape(f,camera_count,n).amin(1)
        # Invalid groups fill unused slots with nearby times, never outrank an
        # observed group. Those slots remain invisible to learned weighting.
        scores = torch.where(group_valid,group_distance,fallback_distance+1e6)
        groups = scores.topk(frame_count,dim=0,largest=False).indices.sort(dim=0).values
        view_ids = (groups[:,None]*camera_count+camera_slots).reshape(-1,n)
        point_ids = torch.arange(n,device=images.device).view(1,-1)
        selected_uv = uv[view_ids,point_ids]
        visible = valid[view_ids,point_ids]
        selected_difference = difference[view_ids,point_ids]
        selected_distance = distance[view_ids,point_ids].masked_fill(~visible,float('inf'))
        source_slot = selected_distance.argmin(dim=0)
        source_view = view_ids[source_slot,point_ids.flatten()]
        source_uv = uv[source_view,point_ids.flatten()]
        rgb = sample_selected_views(flat_rgb,(height,width),source_view[None],
                                    source_uv[None],(height,width)).squeeze(0)
        anchor = ego_to_world[groups[0],:3,3]
        directions = torch.nn.functional.normalize(xyz-origins[source_view],dim=-1)
        moments = torch.cross(origins[source_view]-anchor,directions,dim=-1)
        queries = torch.cat((xyz-anchor,rgb,directions,moments),dim=-1)
        selected_frames = frame_numbers[groups]
        times = ((selected_frames-selected_frames[0]) /
                 (selected_frames[-1]-selected_frames[0]).clamp_min(1)).to(xyz.dtype)
        view_times = times[:,None].expand(-1,camera_count,-1).reshape(-1,n)
        query_times = view_times[source_slot,point_ids.flatten()]
        sampled_features = sample_selected_views(flat_features,feature_shape,view_ids,
                                                selected_uv,(height,width))
        sampled_features *= visible.unsqueeze(-1)
        selected_difference = torch.where(visible,selected_difference,
                                         torch.zeros_like(selected_difference))
        fused, _ = model.fusion.fuse_evidence(queries,sampled_features,visible,
                                             selected_difference,view_times,query_times)
        predicted = model.decoder(fused,queries[:,:3])["color"]
        seen = visible.any(0)
        previous = gaussians.colors[lo:hi]
        gaussians.colors[lo:hi] = torch.where(seen[:,None],
                                             torch.lerp(previous,predicted,alpha),previous)
        updated += int(seen.sum())
        reliable_slots += int(visible.sum())
        if lo//chunk_size % 32 == 0:
            print(f"static-view-bank {hi}/{gaussians.count} updated={updated}",flush=True)
    torch.cuda.synchronize() if gaussians.means.is_cuda else None
    return dict(policy="nearest-depth-consistent-frame-groups-feature-only",
                seconds=time.perf_counter()-start,point_count=gaussians.count,
                updated_count=updated,unseen_retained_count=gaussians.count-updated,
                reliable_view_slots=reliable_slots,bank_frames=frame_numbers.tolist(),
                selected_frame_count=frame_count,selected_view_count=frame_count*camera_count,
                depth_tolerance_m=depth_tolerance,alpha=alpha,chunk_size=chunk_size,
                geometry_opacity_covariance_unchanged=True)
