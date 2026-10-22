"""PointForward: feedforward driving-scene reconstruction (full model).

Wires together the five stages of Fig. 2:

    1. Sec. 3.1  frozen depth + camera geometry  ->  dense point-aligned queries
    2. Sec. 3.2  OmniRe-style scene graph        ->  static / per-instance nodes,
                 dynamic queries canonicalised, then the set is sparsified
    3. Sec. 3.3  spatial-temporal fusion         ->  fused query features + sigma
    4. Sec. 3.4  Gaussian decoding               ->  feature-carrying Gaussians
    5. Sec. 3.4  gsplat rasterisation + UNet     ->  RGB at any target view/time

The model operates on ONE scene per forward call.  Batching is handled by the
training loop, which accumulates gradients over scenes -- the per-scene query
count (V*N_s + O*N_d) is already large enough to saturate a GPU, and each scene
carries its own instance count, so a ragged batch dimension would buy nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import DEFAULT_BACKBONE, FeatureExtractor, sample_features
from .fusion import SpatialTemporalFusion, box_dynamic_masks, project_points
from .gaussian_head import GaussianDecoder, temporal_opacity
from .query import QUERY_DIM, build_dense_queries, plucker, query_xyz, set_query_xyz
from .renderer import FeatureToRGB, render_features
from .rotation import matrix_to_quaternion, quaternion_multiply
from .scene_graph import (
    HUMAN_ID,
    HumanTracks,
    InstanceBoxes,
    assign_instances,
    canonicalise,
    decanonicalise,
    sample_queries,
)
from .smpl import NUM_VERTICES, SMPL


@dataclass
class SceneInput:
    """The V context views of one scene.

    Shapes, with V context views at resolution H x W:
        images       (V, 3, H, W)   RGB in [0, 1]
        intrinsics   (V, 3, 3)      pinhole K, in pixels of the given resolution
        cam_to_world (V, 4, 4)
        times        (V,)           timestamps normalised to [0, 1] over the clip
        frame_index  (V,)           long; which box slot is contemporaneous with
                                    each view (several views share a frame in a
                                    multi-camera rig)
        boxes        scene-graph boxes, or None for a fully static scene
        sparse_depth (V, 1, H, W)   optional LiDAR depth, zeros where missing
        sparse_mask  (V, 1, H, W)   optional validity mask for sparse_depth
        humans       SMPL human nodes on the same slot grid as ``boxes``, or
                     None; only used by a model built with ``use_smpl``
    """

    images: Tensor
    intrinsics: Tensor
    cam_to_world: Tensor
    times: Tensor
    frame_index: Tensor
    boxes: InstanceBoxes | None = None
    sparse_depth: Tensor | None = None
    sparse_mask: Tensor | None = None
    humans: HumanTracks | None = None

    @property
    def num_views(self) -> int:
        return self.images.shape[0]

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])


@dataclass
class TargetViews:
    """The views to render.

    Shapes, with T target views:
        intrinsics   (T, 3, 3)
        cam_to_world (T, 4, 4)
        times        (T,)      normalised render timestamps t'
        frame_index  (T,)      long; box slot to pose dynamic instances at
        height/width           render resolution; defaults to the context size
    """

    intrinsics: Tensor
    cam_to_world: Tensor
    times: Tensor
    frame_index: Tensor
    height: int | None = None
    width: int | None = None

    @property
    def num_views(self) -> int:
        return self.intrinsics.shape[0]


@dataclass
class Gaussians:
    """Decoded Gaussians, still in their own reference frames.

    Static Gaussians live in world space; dynamic ones live in the canonical
    space of their instance and are posed at render time.
    """

    means: Tensor  # (N, 3)
    quats: Tensor  # (N, 4)
    scales: Tensor  # (N, 3)
    opacities: Tensor  # (N,)
    colors: Tensor  # (N, N_c)
    sigma: Tensor  # (N,)
    times: Tensor  # (N,) the query's own timestamp t
    instance_id: Tensor  # (N,) -1 for static
    extras: dict[str, Tensor] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return self.means.shape[0]


class PointForward(nn.Module):
    """The full PointForward model.

    Args:
        depth_model:  a frozen depth predictor (see :mod:`.depth`); must map
                      (V, 3, H, W) -> (V, H, W).
        backbone:     timm name of the image backbone.  The paper uses
                      EfficientNetV2-S.
        feat_dim:     channel width of the image features.
        color_dim:    N_c, the Gaussian colour-feature width (128 in the paper).
        hidden:       hidden width of the fusion MLPs.
        num_static:   N_s, static queries sampled per view.
        num_dynamic:  N_d, dynamic queries sampled per instance.
        num_views:    V, the number of context views (frames x cameras).  Needed
                      up front so the fusion module's per-view head exists before
                      the optimiser is built; 0 defers it, which is fine for
                      inference but would leave it untrained.
        depth_scale:  resolution multiplier for the depth field.  InfiniDepth is
                      resolution-free, so depth can be queried above the feature
                      resolution; 1.0 keeps it at the image resolution.
        use_weighted_fusion / use_depth_cue: the two fusion ablations of Table 4.
        use_smpl:     model SMPL-tracked pedestrians (``SceneInput.humans``) as
                      human nodes with their own decoder.  Off by default, which
                      leaves the model and its state dict unchanged.
        smpl_model_path: ``.npz`` from ``scripts/convert_smpl_model.py``.
    """

    def __init__(
        self,
        depth_model: nn.Module,
        backbone: str = DEFAULT_BACKBONE,
        feat_dim: int = 128,
        color_dim: int = 128,
        hidden: int = 128,
        num_static: int = 100_000,
        num_dynamic: int = 10_000,
        num_views: int = 0,
        depth_scale: float = 1.0,
        use_weighted_fusion: bool = True,
        use_depth_cue: bool = True,
        pretrained_backbone: bool = True,
        fusion_chunk: int = 65_536,
        decoder_chunk: int = 262_144,
        grad_checkpoint: bool = True,
        use_smpl: bool = False,
        smpl_model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.num_static = num_static
        self.num_dynamic = num_dynamic
        self.depth_scale = depth_scale

        self.depth_model = depth_model
        self.backbone = FeatureExtractor(
            out_channels=feat_dim, backbone=backbone, pretrained=pretrained_backbone
        )
        self.fusion = SpatialTemporalFusion(
            feat_dim=feat_dim,
            hidden=hidden,
            query_dim=QUERY_DIM,
            num_views=num_views,
            use_weighted_fusion=use_weighted_fusion,
            use_depth_cue=use_depth_cue,
            chunk_size=fusion_chunk,
            grad_checkpoint=grad_checkpoint,
        )
        self.decoder = GaussianDecoder(
            in_dim=self.fusion.out_dim,
            color_dim=color_dim,
            hidden=hidden * 2,
            chunk_size=decoder_chunk,
            grad_checkpoint=grad_checkpoint,
        )
        self.refiner = FeatureToRGB(in_channels=color_dim)

        self.use_smpl = use_smpl
        self.smpl: SMPL | None = None
        self.human_decoder: GaussianDecoder | None = None
        if use_smpl:
            if smpl_model_path is None:
                raise ValueError("use_smpl needs smpl_model_path")
            self.smpl = SMPL(smpl_model_path)
            # Human Gaussians sit on template vertices ~2 cm apart, so offsets
            # (taken in the rest pose) and scales are bounded far tighter than
            # for depth-lifted queries.
            self.human_decoder = GaussianDecoder(
                in_dim=self.fusion.out_dim,
                color_dim=color_dim,
                hidden=hidden * 2,
                max_offset=0.05,
                scale_init=0.01,
                max_scale=0.1,
                chunk_size=decoder_chunk,
                grad_checkpoint=grad_checkpoint,
            )

    def _humans(self, scene: SceneInput) -> HumanTracks | None:
        """The scene's human nodes, if this model uses them and there are any."""
        humans = scene.humans if self.use_smpl else None
        return humans if humans is not None and humans.num_humans > 0 else None

    # ------------------------------------------------------------------ stages

    @torch.no_grad()
    def predict_depth(self, scene: SceneInput) -> Tensor:
        """Stage 1a: dense depth from the frozen predictor. Returns (V, H, W)."""
        h, w = scene.image_size
        dh, dw = int(round(h * self.depth_scale)), int(round(w * self.depth_scale))
        depth = self.depth_model(
            scene.images,
            height=dh,
            width=dw,
            sparse_depth=scene.sparse_depth,
            sparse_mask=scene.sparse_mask,
        )
        if (dh, dw) != (h, w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
        return depth

    def build_queries(self, scene: SceneInput, depth: Tensor) -> dict[str, Tensor]:
        """Stages 1b + 2: lift pixels to queries, split by scene-graph node, sparsify.

        Returns a dict with the sampled query set:
            queries      (N, 12)  coordinate slice already canonicalised
            world_xyz    (N, 3)   world position at the query's own timestamp
            base_xyz     (N, 3)   frame the Gaussian centre is predicted in
            view_id      (N,)     source view
            instance_id  (N,)     -1 for static
            times        (N,)     query timestamp t
            box_index    (N,)     box slot contemporaneous with the query
        """
        dense, view_id = build_dense_queries(
            scene.images, depth, scene.intrinsics, scene.cam_to_world, z_depth=True
        )
        box_index = scene.frame_index[view_id]

        boxes = scene.boxes
        if boxes is None or boxes.num_instances == 0:
            instance_id = torch.full_like(view_id, -1)
            num_instances = 0
        else:
            instance_id = assign_instances(query_xyz(dense), boxes, box_index)
            num_instances = boxes.num_instances

        humans = self._humans(scene)
        if humans is not None:
            # Pixels on a human belong to its SMPL node, which brings its own
            # template queries; they must not also become static queries.
            on_human = assign_instances(query_xyz(dense), humans.boxes, box_index) >= 0
            instance_id = torch.where(on_human, torch.full_like(instance_id, HUMAN_ID), instance_id)

        idx = sample_queries(
            instance_id=instance_id,
            view_id=view_id,
            num_static_per_view=self.num_static,
            num_dynamic_per_instance=self.num_dynamic,
            num_views=scene.num_views,
            num_instances=num_instances,
        )

        queries = dense[idx]
        view_id = view_id[idx]
        instance_id = instance_id[idx]
        box_index = box_index[idx]
        world_xyz = query_xyz(queries)
        times = scene.times[view_id]

        # Canonicalise dynamic queries (Eq. 2); static queries keep world coords.
        base_xyz = world_xyz
        if boxes is not None and boxes.num_instances > 0:
            base_xyz = self._canonicalise_queries(world_xyz, instance_id, box_index, boxes)

        return {
            "queries": set_query_xyz(queries, base_xyz),
            "world_xyz": world_xyz,
            "base_xyz": base_xyz,
            "view_id": view_id,
            "instance_id": instance_id,
            "times": times,
            "box_index": box_index,
        }

    def _canonicalise_queries(
        self, world_xyz: Tensor, instance_id: Tensor, box_index: Tensor, boxes: InstanceBoxes
    ) -> Tensor:
        """Map dynamic queries into their instance's canonical space."""
        dynamic = instance_id >= 0
        out = world_xyz.clone()
        if not bool(dynamic.any()):
            return out

        inst = instance_id[dynamic]
        slot = box_index[dynamic].clamp(0, boxes.centers.shape[1] - 1)
        can_c, can_r = boxes.canonical()

        out[dynamic] = canonicalise(
            world_xyz[dynamic],
            boxes.centers[inst, slot],
            boxes.rots[inst, slot],
            can_c[inst],
            can_r[inst],
        )
        return out

    def _pose_points(
        self,
        base_xyz: Tensor,
        instance_id: Tensor,
        boxes: InstanceBoxes | None,
        frame: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Place queries in world space at the box poses of the given frames.

        Args:
            base_xyz:    (N, 3) canonical (dynamic) / world (static) coordinates.
            instance_id: (N,)
            boxes:       scene-graph boxes, or None.
            frame:       (K,) long box slots, one per output set.

        Returns:
            points (K, N, 3) world positions.
            valid  (K, N)    False where the instance is absent from that slot.
        """
        k = frame.shape[0]
        n = base_xyz.shape[0]
        points = base_xyz.unsqueeze(0).expand(k, n, 3).clone()
        valid = torch.ones(k, n, dtype=torch.bool, device=base_xyz.device)

        if boxes is None or boxes.num_instances == 0:
            return points, valid

        dynamic = instance_id >= 0
        if not bool(dynamic.any()):
            return points, valid

        inst = instance_id[dynamic]                       # (D,)
        d = inst.shape[0]
        can_c, can_r = boxes.canonical()
        slots = frame.clamp(0, boxes.centers.shape[1] - 1)  # (K,)

        # Gather every (frame, instance) box pair at once.  Looping over frames
        # would cost a GPU sync each, and there is one loop per forward pass.
        inst_k = inst.view(1, d).expand(k, d)
        slot_k = slots.view(k, 1).expand(k, d)
        box_c = boxes.centers[inst_k, slot_k]             # (K, D, 3)
        box_r = boxes.rots[inst_k, slot_k]                # (K, D, 3, 3)

        posed = decanonicalise(
            base_xyz[dynamic].unsqueeze(0).expand(k, d, 3),
            box_c,
            box_r,
            can_c[inst].unsqueeze(0),
            can_r[inst].unsqueeze(0),
        )

        idx = torch.nonzero(dynamic, as_tuple=False).flatten()
        points[:, idx] = posed
        valid[:, idx] = boxes.valid[inst_k, slot_k]
        return points, valid

    def decode(self, scene: SceneInput, depth: Tensor, sampled: dict[str, Tensor]) -> Gaussians:
        """Stages 3 + 4: fuse multi-view evidence, then decode Gaussians."""
        h, w = scene.image_size
        features = self.backbone(scene.images)

        world_to_cam = torch.linalg.inv(scene.cam_to_world)

        points_per_view, point_valid = self._pose_points(
            sampled["base_xyz"], sampled["instance_id"], scene.boxes, scene.frame_index
        )

        dynamic_masks = None
        if scene.boxes is not None and scene.boxes.num_instances > 0:
            dynamic_masks = box_dynamic_masks(
                scene.boxes, scene.frame_index, scene.intrinsics, world_to_cam, h, w
            )
        humans = self._humans(scene)
        if humans is not None:
            human_masks = box_dynamic_masks(
                humans.boxes, scene.frame_index, scene.intrinsics, world_to_cam, h, w
            )
            dynamic_masks = human_masks if dynamic_masks is None else dynamic_masks | human_masks

        fused, sigma = self.fusion(
            queries=sampled["queries"],
            points_per_view=points_per_view,
            features=features,
            depth_maps=depth,
            intrinsics=scene.intrinsics,
            world_to_cam=world_to_cam,
            view_times=scene.times,
            query_times=sampled["times"],
            image_size=(h, w),
            dynamic_masks=dynamic_masks,
            is_static=sampled["instance_id"] < 0,
            point_valid=point_valid,
        )

        decoded = self.decoder(fused, sampled["base_xyz"])
        gaussians = Gaussians(
            means=decoded["means"],
            quats=decoded["quats"],
            scales=decoded["scales"],
            opacities=decoded["opacities"],
            colors=decoded["color"],
            sigma=sigma,
            times=sampled["times"],
            instance_id=sampled["instance_id"],
        )
        if humans is not None:
            gaussians.extras.update(self._decode_humans(scene, depth, features, world_to_cam))
        return gaussians

    def _decode_humans(
        self, scene: SceneInput, depth: Tensor, features: Tensor, world_to_cam: Tensor
    ) -> dict[str, Tensor]:
        """SMPL human nodes: template queries -> fusion -> the human decoder.

        Every human contributes one query per SMPL vertex.  Its coordinate is the
        rest-pose vertex, and it is skinned into each context view's frame to
        gather that view's evidence, just as rigid queries are posed by their
        boxes.  The query's RGB, ray and timestamp come from its anchor view: the
        view that sees the posed vertex with the smallest depth disagreement.

        Returns the canonical (rest-pose) Gaussians under ``human_*`` keys; they
        are skinned to the target frame in :meth:`pose_gaussians`.
        """
        humans = scene.humans
        h, w = scene.image_size
        v, p, n = scene.num_views, humans.num_humans, NUM_VERTICES
        m = p * n

        # Posing and anchoring handle world coordinates, so they stay in fp32.
        with torch.no_grad(), torch.autocast(device_type=depth.device.type, enabled=False):
            slots = scene.frame_index.clamp(0, humans.pose.shape[1] - 1)
            rest, rots, trans = self.smpl.pose_to_world(
                humans.betas,
                humans.pose[:, slots].transpose(0, 1),
                humans.boxes.centers[:, slots].transpose(0, 1),
            )
            rest = rest.reshape(m, 3)
            points = (rots.reshape(v, m, 3, 3) @ rest.unsqueeze(-1)).squeeze(-1) + trans.reshape(v, m, 3)
            point_valid = (
                humans.pose_valid[:, slots].transpose(0, 1).unsqueeze(-1).expand(v, p, n).reshape(v, m)
            )

            uv, proj_depth, in_front = project_points(points, scene.intrinsics, world_to_cam)
            in_bounds = (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
            seen = in_front & in_bounds & point_valid
            ref_depth = sample_features(depth.unsqueeze(1), uv, h, w).squeeze(-1)
            error = torch.where(seen, (proj_depth - ref_depth).abs(), torch.full_like(proj_depth, float("inf")))
            # Vertices no view sees fall back to view 0; fusion gives them no
            # evidence either way.
            anchor = error.argmin(dim=0)
            rows = torch.arange(m, device=anchor.device)

            rgb = sample_features(scene.images, uv, h, w)[anchor, rows]
            origins = scene.cam_to_world[anchor, :3, 3]
            directions = F.normalize(points[anchor, rows] - origins, dim=-1)
            queries = torch.cat((rest, rgb, plucker(origins, directions)), dim=-1)
            times = scene.times[anchor]

        fused, _ = self.fusion(
            queries=queries,
            points_per_view=points,
            features=features,
            depth_maps=depth,
            intrinsics=scene.intrinsics,
            world_to_cam=world_to_cam,
            view_times=scene.times,
            query_times=times,
            image_size=(h, w),
            point_valid=point_valid,
        )
        decoded = self.human_decoder(fused, rest)
        return {
            "human_means": decoded["means"],
            "human_quats": decoded["quats"],
            "human_scales": decoded["scales"],
            "human_opacities": decoded["opacities"],
            "human_colors": decoded["color"],
        }

    def pose_gaussians(
        self,
        gaussians: Gaussians,
        boxes: InstanceBoxes | None,
        frame: int,
        target_time: Tensor | float,
        humans: HumanTracks | None = None,
    ) -> dict[str, Tensor]:
        """Push Gaussians into world space at one target frame / timestamp.

        Dynamic Gaussians are de-canonicalised, which also rotates their
        orientation; opacity is then attenuated by Eq. (7).  Human Gaussians,
        when ``humans`` is given and the Gaussians carry them, are skinned to the
        frame's SMPL pose instead and appended; Eq. (7) does not apply to them.
        """
        means = gaussians.means
        quats = gaussians.quats
        opacities = temporal_opacity(
            gaussians.opacities, gaussians.sigma, gaussians.times, target_time
        )

        if boxes is not None and boxes.num_instances > 0:
            dynamic = gaussians.instance_id >= 0
            if bool(dynamic.any()):
                inst = gaussians.instance_id[dynamic]
                slot = min(frame, boxes.centers.shape[1] - 1)
                box_c = boxes.centers[inst, slot]
                box_r = boxes.rots[inst, slot]
                can_c, can_r = boxes.canonical()

                means = means.clone()
                quats = quats.clone()
                opacities = opacities.clone()

                means[dynamic] = decanonicalise(
                    gaussians.means[dynamic], box_c, box_r, can_c[inst], can_r[inst]
                )
                # Column-vector convention: the linear part of the rigid map is
                # R_box^T R_can, and it composes on the left of the orientation.
                rel = torch.einsum("nji,njk->nik", box_r, can_r[inst])
                quats[dynamic] = quaternion_multiply(
                    matrix_to_quaternion(rel), gaussians.quats[dynamic]
                )
                # Instances absent from this frame contribute nothing.
                absent = ~boxes.valid[inst, slot]
                if bool(absent.any()):
                    dyn_idx = torch.nonzero(dynamic, as_tuple=False).flatten()
                    opacities[dyn_idx[absent]] = 0.0

        posed = {
            "means": means,
            "quats": quats,
            "scales": gaussians.scales,
            "opacities": opacities,
            "colors": gaussians.colors,
        }
        if humans is not None and "human_means" in gaussians.extras:
            posed = {k: torch.cat((posed[k], h.to(posed[k].dtype)), dim=0)
                     for k, h in self._pose_humans(gaussians.extras, humans, frame).items()}
        return posed

    def _pose_humans(self, extras: dict[str, Tensor], humans: HumanTracks, frame: int) -> dict[str, Tensor]:
        """Skin the canonical human Gaussians to one frame's SMPL poses."""
        slot = min(frame, humans.pose.shape[1] - 1)
        m = humans.num_humans * NUM_VERTICES
        with torch.no_grad():
            _, rots, trans = self.smpl.pose_to_world(
                humans.betas, humans.pose[None, :, slot], humans.boxes.centers[None, :, slot]
            )
            rots, trans = rots.reshape(m, 3, 3), trans.reshape(m, 3)
            present = humans.pose_valid[:, slot].repeat_interleave(NUM_VERTICES)

        # World-space placement in fp32; bf16 would quantise it to ~0.2 m.
        with torch.autocast(device_type=rots.device.type, enabled=False):
            means = (rots @ extras["human_means"].float().unsqueeze(-1)).squeeze(-1) + trans
            quats = quaternion_multiply(matrix_to_quaternion(rots), extras["human_quats"].float())
        return {
            "means": means,
            "quats": quats,
            "scales": extras["human_scales"],
            # A human without a pose at this frame contributes nothing.
            "opacities": extras["human_opacities"] * present.to(extras["human_opacities"].dtype),
            "colors": extras["human_colors"],
        }

    def render(
        self,
        gaussians: Gaussians,
        boxes: InstanceBoxes | None,
        targets: TargetViews,
        height: int,
        width: int,
        render_depth: bool = False,
        humans: HumanTracks | None = None,
    ) -> dict[str, Tensor]:
        """Stage 5: rasterise features per target view, then refine to RGB.

        Target views sharing a frame index are rasterised in a single gsplat
        call, since the dynamic Gaussian poses are identical for them.  Human
        Gaussians join the same call when ``humans`` is given.
        """
        t = targets.num_views
        world_to_cam = torch.linalg.inv(targets.cam_to_world)

        feats_out: list[Tensor] = []
        alpha_out: list[Tensor] = []
        depth_out: list[Tensor] = []
        order: list[Tensor] = []

        frames = targets.frame_index.tolist()
        for frame in sorted(set(frames)):
            sel = torch.tensor(
                [i for i, f in enumerate(frames) if f == frame],
                dtype=torch.long,
                device=gaussians.means.device,
            )
            # Views in a frame group share a timestamp by construction.
            posed = self.pose_gaussians(gaussians, boxes, int(frame), targets.times[sel[0]], humans)
            feats, alpha, depth = render_features(
                means=posed["means"],
                quats=posed["quats"],
                scales=posed["scales"],
                opacities=posed["opacities"],
                colors=posed["colors"],
                intrinsics=targets.intrinsics[sel],
                world_to_cam=world_to_cam[sel],
                height=height,
                width=width,
                render_depth=render_depth,
            )
            feats_out.append(feats)
            alpha_out.append(alpha)
            if depth is not None:
                depth_out.append(depth)
            order.append(sel)

        features = torch.cat(feats_out, dim=0)
        alpha = torch.cat(alpha_out, dim=0)

        # Restore the caller's target ordering.  Advanced indexing
        # (features[perm]) triggers a "CUDA error: invalid configuration
        # argument" from PyTorch's generic indexing kernel once the tensor
        # is large enough (observed at (12, 128, 700, 1036), ~1.1e9
        # elements, with torch 2.8.0+cu128) -- index_select along dim 0 is
        # semantically identical for a 1-D index and does not hit this limit.
        perm = torch.argsort(torch.cat(order, dim=0))
        features = torch.index_select(features, 0, perm)
        alpha = torch.index_select(alpha, 0, perm)

        rgb = self.refiner(features, alpha)
        assert rgb.shape[0] == t
        out = {"rgb": rgb, "features": features, "alpha": alpha}
        if depth_out:
            depth_cat = torch.cat(depth_out, dim=0)
            out["render_depth"] = torch.index_select(depth_cat, 0, perm)
        return out

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        scene: SceneInput,
        targets: TargetViews,
        return_gaussians: bool = False,
        render_depth: bool = False,
    ) -> dict[str, Tensor]:
        """Reconstruct the scene and render the target views.

        Returns a dict with ``rgb`` (T, 3, H, W), ``alpha`` (T, 1, H, W) and
        ``depth`` (V, H, W) -- the *input* depth prior.  ``render_depth=True``
        additionally returns ``render_depth`` (T, 1, H, W), the rasterised
        expected depth that the D-RMSE metric is computed against.
        """
        depth = self.predict_depth(scene)
        sampled = self.build_queries(scene, depth)
        gaussians = self.decode(scene, depth, sampled)

        h, w = scene.image_size
        out = self.render(
            gaussians,
            scene.boxes,
            targets,
            height=targets.height or h,
            width=targets.width or w,
            render_depth=render_depth,
            humans=self._humans(scene),
        )
        out["depth"] = depth
        if return_gaussians:
            out["gaussians"] = gaussians
        return out
