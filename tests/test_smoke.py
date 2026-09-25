"""Shape / gradient smoke tests and geometry unit checks.

Run with:
    pytest -q tests/test_smoke.py

The rasterisation stage needs gsplat and a CUDA device, so those tests skip on
CPU-only machines; everything upstream of it runs anywhere.
"""

from __future__ import annotations

import math

import pytest
import torch

from pointforward.losses import RenderLoss, psnr, ssim
from pointforward.models import (
    ConstantDepth,
    InstanceBoxes,
    PointForward,
    SceneInput,
    TargetViews,
    assign_instances,
    canonicalise,
    decanonicalise,
    inside_box,
    matrix_to_quaternion,
    quaternion_to_matrix,
    sample_queries,
)

torch.manual_seed(0)

V, H, W = 3, 48, 64
NUM_INSTANCES, NUM_FRAMES = 2, 3


def _yaw(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _boxes() -> InstanceBoxes:
    centers = torch.zeros(NUM_INSTANCES, NUM_FRAMES, 3)
    dims = torch.zeros(NUM_INSTANCES, NUM_FRAMES, 3)
    rots = torch.zeros(NUM_INSTANCES, NUM_FRAMES, 3, 3)
    times = torch.zeros(NUM_INSTANCES, NUM_FRAMES)
    valid = torch.ones(NUM_INSTANCES, NUM_FRAMES, dtype=torch.bool)

    for o in range(NUM_INSTANCES):
        for j in range(NUM_FRAMES):
            # Instances translate along +x and rotate slightly per frame.
            centers[o, j] = torch.tensor([2.0 + o * 3.0 + j * 0.7, 1.0 + o, 8.0])
            dims[o, j] = torch.tensor([4.5, 2.0, 1.8])
            rots[o, j] = _yaw(0.15 * j + 0.4 * o)
            times[o, j] = j / (NUM_FRAMES - 1)

    # Instance 1 enters the scene only at frame 1, so slot 0 is invalid: this
    # exercises canonical_index() picking the earliest *valid* box.
    valid[1, 0] = False
    return InstanceBoxes(centers=centers, dims=dims, rots=rots, times=times, valid=valid)


def _scene(device: torch.device) -> SceneInput:
    focal = 0.9 * W
    k = torch.tensor([[focal, 0.0, W / 2], [0.0, focal, H / 2], [0.0, 0.0, 1.0]])
    intrinsics = k.unsqueeze(0).repeat(V, 1, 1)

    cam_to_world = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
    for v in range(V):
        cam_to_world[v, 0, 3] = v * 0.4  # small lateral baseline

    return SceneInput(
        images=torch.rand(V, 3, H, W, device=device),
        intrinsics=intrinsics.to(device),
        cam_to_world=cam_to_world.to(device),
        times=torch.linspace(0, 1, V, device=device),
        frame_index=torch.arange(V, device=device) % NUM_FRAMES,
        boxes=_move_boxes(_boxes(), device),
        sparse_depth=None,
        sparse_mask=None,
    )


def _move_boxes(boxes: InstanceBoxes, device: torch.device) -> InstanceBoxes:
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


def _targets(scene: SceneInput) -> TargetViews:
    return TargetViews(
        intrinsics=scene.intrinsics,
        cam_to_world=scene.cam_to_world,
        times=scene.times,
        frame_index=scene.frame_index,
    )


def _model(device: torch.device) -> PointForward:
    return PointForward(
        depth_model=ConstantDepth(8.0),
        feat_dim=32,
        color_dim=32,
        hidden=32,
        num_static=512,
        num_dynamic=128,
        num_views=V,
        pretrained_backbone=False,
    ).to(device)


# ------------------------------------------------------------------- geometry


def test_canonicalise_round_trip():
    boxes = _boxes()
    can_c, can_r = boxes.canonical()
    points = torch.randn(64, 3) * 2.0 + boxes.centers[0, 1]

    bar = canonicalise(points, boxes.centers[0, 1], boxes.rots[0, 1], can_c[0], can_r[0])
    back = decanonicalise(bar, boxes.centers[0, 1], boxes.rots[0, 1], can_c[0], can_r[0])
    assert torch.allclose(points, back, atol=1e-5)


def test_canonical_index_skips_invalid_slots():
    boxes = _boxes()
    idx = boxes.canonical_index()
    assert idx.tolist() == [0, 1]  # instance 1 is absent at frame 0

    can_c, _ = boxes.canonical()
    assert torch.allclose(can_c[1], boxes.centers[1, 1])


def test_canonicalisation_is_rigid():
    """Distances between points must survive the canonical mapping."""
    boxes = _boxes()
    can_c, can_r = boxes.canonical()
    p = torch.randn(32, 3) * 3.0
    q = torch.randn(32, 3) * 3.0

    args = (boxes.centers[0, 2], boxes.rots[0, 2], can_c[0], can_r[0])
    d_before = (p - q).norm(dim=-1)
    d_after = (canonicalise(p, *args) - canonicalise(q, *args)).norm(dim=-1)
    assert torch.allclose(d_before, d_after, atol=1e-5)


def test_inside_box():
    center = torch.tensor([1.0, 2.0, 3.0])
    dims = torch.tensor([2.0, 4.0, 6.0])
    rot = _yaw(0.0)

    assert bool(inside_box(center, center, dims, rot))
    # Just inside / just outside along each axis.
    assert bool(inside_box(center + torch.tensor([0.99, 0.0, 0.0]), center, dims, rot))
    assert not bool(inside_box(center + torch.tensor([1.01, 0.0, 0.0]), center, dims, rot))
    assert not bool(inside_box(center + torch.tensor([0.0, 0.0, 3.01]), center, dims, rot))


def test_inside_box_respects_rotation():
    center = torch.zeros(3)
    dims = torch.tensor([8.0, 1.0, 1.0])  # long along local x
    p = torch.tensor([0.0, 3.0, 0.0])

    assert not bool(inside_box(p, center, dims, _yaw(0.0)))
    # Rotating the box 90 degrees brings the point inside.
    assert bool(inside_box(p, center, dims, _yaw(math.pi / 2)))


def test_assign_instances_matches_boxes():
    boxes = _boxes()
    # One point at each instance's frame-2 centre, plus one far away.
    points = torch.stack((boxes.centers[0, 2], boxes.centers[1, 2], torch.tensor([0.0, 0.0, -50.0])))
    box_index = torch.full((3,), 2, dtype=torch.long)

    assert assign_instances(points, boxes, box_index).tolist() == [0, 1, -1]


def test_assign_instances_chunking_is_equivalent():
    boxes = _boxes()
    points = torch.randn(5000, 3) * 6.0
    box_index = torch.randint(NUM_FRAMES, (5000,))

    whole = assign_instances(points, boxes, box_index, chunk=10_000)
    chunked = assign_instances(points, boxes, box_index, chunk=97)
    assert torch.equal(whole, chunked)


def test_assign_instances_ignores_invalid_slots():
    boxes = _boxes()
    # Instance 1 has no box at frame 0, so its frame-0 centre must read static.
    points = boxes.centers[1, 0].unsqueeze(0)
    assert assign_instances(points, boxes, torch.zeros(1, dtype=torch.long)).tolist() == [-1]


def test_sample_queries_exact_count():
    instance_id = torch.full((4000,), -1, dtype=torch.long)
    instance_id[:50] = 0  # instance 1 gets no queries at all
    view_id = torch.randint(V, (4000,))

    idx = sample_queries(
        instance_id=instance_id,
        view_id=view_id,
        num_static_per_view=64,
        num_dynamic_per_instance=32,
        num_views=V,
        num_instances=NUM_INSTANCES,
    )
    assert idx.shape == (V * 64 + NUM_INSTANCES * 32,)
    assert int(idx.max()) < 4000


def test_sample_queries_explicit_generator_is_reproducible():
    instance_id = torch.full((4000,), -1, dtype=torch.long)
    instance_id[:50] = 0
    view_id = torch.arange(4000) % V

    def sample(seed: int) -> torch.Tensor:
        return sample_queries(
            instance_id=instance_id,
            view_id=view_id,
            num_static_per_view=64,
            num_dynamic_per_instance=32,
            num_views=V,
            num_instances=1,
            generator=torch.Generator().manual_seed(seed),
        )

    first = sample(7)
    assert torch.equal(first, sample(7))
    assert not torch.equal(first, sample(8))


def test_quaternion_matrix_round_trip():
    q = torch.nn.functional.normalize(torch.randn(128, 4), dim=-1)
    back = matrix_to_quaternion(quaternion_to_matrix(q))
    # q and -q denote the same rotation, so compare via the matrices.
    assert torch.allclose(quaternion_to_matrix(back), quaternion_to_matrix(q), atol=1e-5)


def test_quaternion_matrix_round_trip_near_pi():
    """The 180-degree case is where the naive trace formula breaks down."""
    mats = torch.stack((_yaw(math.pi), _yaw(math.pi - 1e-4), _yaw(-math.pi + 1e-4)))
    back = quaternion_to_matrix(matrix_to_quaternion(mats))
    assert torch.allclose(back, mats, atol=1e-4)


# ---------------------------------------------------------------------- stages


def test_query_and_decode_shapes():
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)

    depth = model.predict_depth(scene)
    assert depth.shape == (V, H, W)

    sampled = model.build_queries(scene, depth)
    expected_n = V * model.num_static + NUM_INSTANCES * model.num_dynamic
    assert sampled["queries"].shape == (expected_n, 12)
    assert sampled["base_xyz"].shape == (expected_n, 3)
    assert sampled["times"].shape == (expected_n,)
    assert int(sampled["instance_id"].max()) < NUM_INSTANCES
    # Both scene-graph branches must actually be populated, otherwise the tests
    # below would pass vacuously.
    assert bool((sampled["instance_id"] >= 0).any()), "no query landed in a box"
    assert bool((sampled["instance_id"] < 0).any()), "no static query"

    gaussians = model.decode(scene, depth, sampled)
    assert gaussians.count == expected_n
    assert gaussians.quats.shape == (expected_n, 4)
    assert gaussians.scales.shape == (expected_n, 3)
    assert gaussians.opacities.shape == (expected_n,)
    assert gaussians.colors.shape == (expected_n, 32)
    assert torch.all(gaussians.sigma >= 0)
    assert torch.isfinite(gaussians.means).all()
    assert torch.allclose(gaussians.quats.norm(dim=-1), torch.ones(expected_n), atol=1e-5)


def test_static_queries_keep_world_coordinates():
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    static = sampled["instance_id"] < 0
    assert bool(static.any())
    assert torch.allclose(sampled["base_xyz"][static], sampled["world_xyz"][static])


def test_dynamic_queries_are_canonicalised():
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    dynamic = sampled["instance_id"] >= 0
    if not bool(dynamic.any()):
        pytest.skip("no dynamic queries hit the synthetic boxes")

    # Round-tripping the canonical position through the query's own frame must
    # recover the world position it was lifted from.
    boxes = scene.boxes
    inst = sampled["instance_id"][dynamic]
    slot = sampled["box_index"][dynamic]
    can_c, can_r = boxes.canonical()
    back = decanonicalise(
        sampled["base_xyz"][dynamic],
        boxes.centers[inst, slot],
        boxes.rots[inst, slot],
        can_c[inst],
        can_r[inst],
    )
    assert torch.allclose(back, sampled["world_xyz"][dynamic], atol=1e-4)


def test_temporal_opacity_decays():
    from pointforward.models import temporal_opacity

    o = torch.ones(4)
    sigma = torch.zeros(4)
    t = torch.zeros(4)

    same = temporal_opacity(o, sigma, t, 0.0)
    far = temporal_opacity(o, sigma, t, 3.0)
    assert torch.allclose(same, torch.ones(4))
    assert bool((far < same).all())

    # A larger lifespan decays more slowly.
    slow = temporal_opacity(o, torch.full((4,), 5.0), t, 3.0)
    assert bool((slow > far).all())


def test_fusion_ablations_run():
    device = torch.device("cpu")
    scene = _scene(device)
    for weighted, depth_cue in ((False, True), (True, False), (False, False)):
        model = PointForward(
            depth_model=ConstantDepth(8.0),
            feat_dim=16,
            color_dim=16,
            hidden=16,
            num_static=128,
            num_dynamic=32,
            num_views=V,
            use_weighted_fusion=weighted,
            use_depth_cue=depth_cue,
            pretrained_backbone=False,
        ).to(device)
        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth)
        gaussians = model.decode(scene, depth, sampled)
        assert torch.isfinite(gaussians.means).all()


def test_phi_v_exists_before_first_forward():
    """phi_v must be a registered parameter at construction time.

    An optimiser is built from model.parameters() before any forward pass, so a
    lazily-created phi_v would never be registered with it and would sit frozen
    at its initialisation for the whole run -- while training loss still fell,
    hiding the problem.
    """
    device = torch.device("cpu")
    model = PointForward(
        depth_model=ConstantDepth(8.0),
        feat_dim=16, color_dim=16, hidden=16,
        num_static=64, num_dynamic=16,
        num_views=V,
        pretrained_backbone=False,
    ).to(device)

    names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith("fusion.phi_v") for n in names), "phi_v not registered"
    # And it must appear exactly once -- registering it as both an attribute and
    # a submodule would duplicate it in state_dict.
    keys = [k for k in model.state_dict() if "phi_v" in k]
    assert keys and not any(k.startswith("fusion._phi_v") for k in keys), keys

    before = len(list(model.parameters()))
    scene = _scene(device)
    depth = model.predict_depth(scene)
    model.decode(scene, depth, model.build_queries(scene, depth))
    assert len(list(model.parameters())) == before, "forward pass created new parameters"


def test_phi_v_rejects_wrong_view_count():
    """Silently rebuilding phi_v for a different V would leave it unoptimised."""
    device = torch.device("cpu")
    model = PointForward(
        depth_model=ConstantDepth(8.0),
        feat_dim=16, color_dim=16, hidden=16,
        num_static=64, num_dynamic=16,
        num_views=V + 1,          # deliberately wrong
        pretrained_backbone=False,
    ).to(device)

    scene = _scene(device)
    depth = model.predict_depth(scene)
    with pytest.raises(ValueError, match="num_views"):
        model.decode(scene, depth, model.build_queries(scene, depth))


def test_pose_points_returns_world_positions():
    """Posing a dynamic query at its own frame must recover its world position.

    _pose_points gathers all (frame, instance) box pairs at once rather than
    looping; this checks the gathered indexing lines up, which an off-by-one in
    the expand/gather would silently break.
    """
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    dynamic = sampled["instance_id"] >= 0
    assert bool(dynamic.any())

    points, valid = model._pose_points(
        sampled["base_xyz"], sampled["instance_id"], scene.boxes, scene.frame_index
    )
    assert points.shape == (V, sampled["base_xyz"].shape[0], 3)

    # For each view, the queries whose own frame is that view's frame must land
    # back on the world position they were lifted from.
    for v in range(V):
        same = dynamic & (sampled["box_index"] == int(scene.frame_index[v]))
        if not bool(same.any()):
            continue
        assert torch.allclose(points[v][same], sampled["world_xyz"][same], atol=1e-4)

    # Static queries are frame-independent.
    static = ~dynamic
    for v in range(V):
        assert torch.allclose(points[v][static], sampled["base_xyz"][static])
    assert bool(valid[:, static].all())


def test_box_dynamic_masks_covers_a_known_box():
    """A box straight ahead must mask the image region it projects onto."""
    from pointforward.models.fusion import box_dynamic_masks, project_points

    h, w = 48, 64
    focal = 0.9 * w
    k = torch.tensor([[focal, 0.0, w / 2], [0.0, focal, h / 2], [0.0, 0.0, 1.0]]).unsqueeze(0)
    world_to_cam = torch.eye(4).unsqueeze(0)

    boxes = InstanceBoxes(
        centers=torch.tensor([[[0.0, 0.0, 10.0]]]),
        dims=torch.tensor([[[2.0, 2.0, 2.0]]]),
        rots=torch.eye(3).view(1, 1, 3, 3),
        times=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    mask = box_dynamic_masks(boxes, torch.zeros(1, dtype=torch.long), k, world_to_cam, h, w)

    assert mask.shape == (1, h, w)
    assert bool(mask.any()), "a box in front of the camera produced an empty mask"
    # The box centre projects to the principal point, which must be masked.
    assert bool(mask[0, h // 2, w // 2])
    # A 2 m box at 10 m spans ~ focal * 2 / 10 pixels; the mask must not be the
    # whole image.
    assert float(mask.float().mean()) < 0.5

    # An out-of-frame box contributes nothing.
    behind = InstanceBoxes(
        centers=torch.tensor([[[0.0, 0.0, -10.0]]]),
        dims=boxes.dims, rots=boxes.rots, times=boxes.times, valid=boxes.valid,
    )
    assert not bool(
        box_dynamic_masks(behind, torch.zeros(1, dtype=torch.long), k, world_to_cam, h, w).any()
    )


def test_box_dynamic_masks_skips_invalid_slots():
    from pointforward.models.fusion import box_dynamic_masks

    h, w = 32, 32
    k = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]).unsqueeze(0)
    boxes = InstanceBoxes(
        centers=torch.tensor([[[0.0, 0.0, 8.0]]]),
        dims=torch.tensor([[[2.0, 2.0, 2.0]]]),
        rots=torch.eye(3).view(1, 1, 3, 3),
        times=torch.zeros(1, 1),
        valid=torch.zeros(1, 1, dtype=torch.bool),   # absent at this slot
    )
    mask = box_dynamic_masks(boxes, torch.zeros(1, dtype=torch.long), k, torch.eye(4).unsqueeze(0), h, w)
    assert not bool(mask.any())


def test_fusion_chunking_is_exact():
    """Chunked fusion must match the unchunked result bit-for-bit-ish.

    Every operation in fusion is independent across queries -- the softmax
    reduces over views -- so chunking is an exactness-preserving memory
    optimisation, not an approximation.  This test is what makes that claim
    checkable rather than asserted.
    """
    device = torch.device("cpu")
    scene = _scene(device)
    model = _model(device).eval()

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)

    with torch.no_grad():
        model.fusion.chunk_size = 0  # no chunking
        whole = model.decode(scene, depth, sampled)
        model.fusion.chunk_size = 137  # deliberately awkward, non-divisible
        model.decoder.chunk_size = 91
        chunked = model.decode(scene, depth, sampled)

    assert torch.allclose(whole.sigma, chunked.sigma, atol=1e-5)
    assert torch.allclose(whole.colors, chunked.colors, atol=1e-5)
    assert torch.allclose(whole.means, chunked.means, atol=1e-5)
    assert torch.allclose(whole.opacities, chunked.opacities, atol=1e-5)


def test_grad_checkpointing_matches_plain_backward():
    """Recomputation must not change the gradients it produces."""
    device = torch.device("cpu")
    scene = _scene(device)

    grads: dict[bool, torch.Tensor] = {}
    for ckpt in (False, True):
        torch.manual_seed(1234)
        model = _model(device)
        model.fusion.chunk_size = 97
        model.decoder.chunk_size = 97
        model.fusion.grad_checkpoint = ckpt
        model.decoder.grad_checkpoint = ckpt
        model.train()

        torch.manual_seed(7)  # identical query sampling in both runs
        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth)
        g = model.decode(scene, depth, sampled)
        (g.colors.square().mean() + g.means.square().mean() + g.sigma.mean()).backward()

        grads[ckpt] = torch.cat(
            [p.grad.flatten() for p in model.fusion.parameters() if p.grad is not None]
        )

    assert grads[False].shape == grads[True].shape
    assert torch.allclose(grads[False], grads[True], atol=1e-5)


def test_static_scene_without_boxes():
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)
    scene.boxes = None

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    assert sampled["queries"].shape[0] == V * model.num_static
    assert bool((sampled["instance_id"] < 0).all())
    assert torch.isfinite(model.decode(scene, depth, sampled).means).all()


def test_gradients_reach_trainable_modules():
    device = torch.device("cpu")
    model = _model(device)
    scene = _scene(device)

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    gaussians = model.decode(scene, depth, sampled)

    # Rasterisation needs CUDA, so drive the backward pass from the decoded
    # Gaussians instead -- it covers the backbone, fusion and decoder.
    loss = (
        gaussians.colors.square().mean()
        + gaussians.means.square().mean()
        + gaussians.opacities.mean()
        + gaussians.scales.mean()
        + gaussians.sigma.mean()
    )
    loss.backward()

    for name in ("backbone", "fusion", "decoder"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert grads, f"no gradients reached {name}"
        assert any(bool(g.abs().sum() > 0) for g in grads), f"all-zero gradients in {name}"

    # The frozen depth model must stay frozen.
    assert all(not p.requires_grad for p in model.depth_model.parameters())


# ---------------------------------------------------------------------- losses


def test_render_loss_and_metrics():
    pred = torch.rand(2, 3, 32, 40, requires_grad=True)
    target = torch.rand(2, 3, 32, 40)

    criterion = RenderLoss(lpips_weight=0.0)
    out = criterion(pred, target)
    assert set(out) == {"l1", "loss"}
    out["loss"].backward()
    assert pred.grad is not None

    assert float(psnr(target, target)) > 60.0
    assert pytest.approx(float(ssim(target, target)), abs=1e-4) == 1.0
    assert float(psnr(pred.detach(), target)) < float(psnr(target, target))


def test_render_loss_mask_excludes_pixels():
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    target[:, :, :4] = 1.0  # only the top half differs

    criterion = RenderLoss(lpips_weight=0.0)
    mask = torch.ones(1, 1, 8, 8)
    mask[:, :, :4] = 0.0  # ...and it is masked out
    assert float(criterion(pred, target, mask)["l1"]) == pytest.approx(0.0)
    assert float(criterion(pred, target)["l1"]) > 0.4


# ------------------------------------------------------------------ rendering


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterisation needs CUDA")
def test_full_forward_backward_cuda():
    pytest.importorskip("gsplat")

    device = torch.device("cuda")
    model = _model(device)
    scene = _scene(device)
    targets = _targets(scene)

    out = model(scene, targets)
    assert out["rgb"].shape == (V, 3, H, W)
    assert out["alpha"].shape == (V, 1, H, W)
    assert bool((out["rgb"] >= 0).all() and (out["rgb"] <= 1).all())

    criterion = RenderLoss(lpips_weight=0.0).to(device)
    criterion(out["rgb"], scene.images)["loss"].backward()

    trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert trained
    assert any(bool(p.grad.abs().sum() > 0) for p in trained)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterisation needs CUDA")
def test_render_target_ordering_is_preserved():
    """Views are grouped by frame for rasterisation, then reordered back.

    Renders one fixed Gaussian set twice, so the only difference between the two
    calls is the target ordering -- a full forward would resample its queries and
    the two results would legitimately differ.
    """
    pytest.importorskip("gsplat")

    device = torch.device("cuda")
    model = _model(device)
    scene = _scene(device)
    targets = _targets(scene)

    with torch.no_grad():
        depth = model.predict_depth(scene)
        gaussians = model.decode(scene, depth, model.build_queries(scene, depth))

        perm = torch.tensor([2, 0, 1], device=device)
        shuffled = TargetViews(
            intrinsics=targets.intrinsics[perm],
            cam_to_world=targets.cam_to_world[perm],
            times=targets.times[perm],
            frame_index=targets.frame_index[perm],
        )
        out = model.render(gaussians, scene.boxes, targets, H, W)
        out_shuffled = model.render(gaussians, scene.boxes, shuffled, H, W)

    assert torch.allclose(out["rgb"][perm], out_shuffled["rgb"], atol=1e-5)
