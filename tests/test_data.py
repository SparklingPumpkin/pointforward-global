"""Loader tests that run without the real dataset.

A miniature scene is synthesised on disk in the DriveStudio layout, with values
chosen so the correct answer is known analytically.  This pins down the parts
that are easy to get subtly wrong -- LiDAR column layout, the Waymo/OpenCV axis
convention, the box rotation convention and world recentring -- without needing
the 127 GB corpus.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from pointforward.data.drivestudio import (
    OPENCV2WAYMO,
    WaymoDriveStudioClips,
)
from pointforward.data.argoverse import ArgoverseDriveStudioClips
from pointforward.data import to_scene_input
from pointforward.models.scene_graph import inside_box

CAMS = (0, 1, 2)
N_FRAMES = 24
IMG_H, IMG_W = 64, 96


def _write_lidar(path: str, points: np.ndarray, cols: int) -> None:
    """Write a sweep in the Nx14 or Nx10 layout, points at columns 3:6."""
    n = len(points)
    a = np.zeros((n, cols), dtype=np.float32)
    a[:, 0:3] = [1.5, 0.0, 2.1]          # sensor origin
    a[:, 3:6] = points
    a[:, cols - 4] = np.random.randint(0, 2, n)   # ground label, binary
    a[:, cols - 3] = np.random.rand(n) * 100      # intensity
    a[:, cols - 2] = np.random.rand(n)            # elongation
    a[:, cols - 1] = np.random.randint(0, 5, n)   # laser id in 0..4
    a.tofile(path)


def _make_scene(root: str, name: str, lidar_cols: int | None = 14) -> str:
    """Build one synthetic scene; returns its path."""
    d = os.path.join(root, name)
    for sub in ("images", "ego_pose", "intrinsics", "extrinsics", "lidar", "sky_masks",
                os.path.join("dynamic_masks", "all"), "instances"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)

    rng = np.random.default_rng(0)

    # Calibration: cameras sit at the ego origin with Waymo axes (x forward).
    for cam in CAMS:
        np.savetxt(
            os.path.join(d, "intrinsics", f"{cam}.txt"),
            np.array([100.0, 100.0, IMG_W / 2, IMG_H / 2, 0, 0, 0, 0, 0]),
        )
        ext = np.eye(4)
        ext[:3, 3] = [1.5, 0.0, 2.1]
        np.savetxt(os.path.join(d, "extrinsics", f"{cam}.txt"), ext)

    for f in range(N_FRAMES):
        # Ego drives along world +x, starting far from the origin so recentring
        # has something to do.
        ego = np.eye(4)
        ego[:3, 3] = [10_000.0 + f * 2.0, 5_000.0, 100.0]
        np.savetxt(os.path.join(d, "ego_pose", f"{f:03d}.txt"), ego)

        for cam in CAMS:
            Image.fromarray(
                rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
            ).save(os.path.join(d, "images", f"{f:03d}_{cam}.jpg"))
            Image.fromarray(
                (rng.random((IMG_H, IMG_W)) > 0.7).astype(np.uint8) * 255
            ).save(os.path.join(d, "dynamic_masks", "all", f"{f:03d}_{cam}.png"))
            Image.fromarray(
                np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            ).save(os.path.join(d, "sky_masks", f"{f:03d}_{cam}.png"))

        if lidar_cols is not None:
            # Points in the ego frame, ahead of and around the vehicle.
            pts = np.stack(
                [
                    rng.uniform(5.0, 60.0, 500),    # x forward
                    rng.uniform(-20.0, 20.0, 500),  # y left
                    rng.uniform(-2.0, 4.0, 500),    # z up
                ],
                axis=1,
            )
            _write_lidar(os.path.join(d, "lidar", f"{f:03d}.bin"), pts, lidar_cols)

    # One instance driving ahead of the ego vehicle, in world space.
    ann = {"frame_idx": [], "obj_to_world": [], "box_size": []}
    for f in range(N_FRAMES):
        pose = np.eye(4)
        pose[:3, 3] = [10_030.0 + f * 3.0, 5_000.0, 100.8]
        ann["frame_idx"].append(f)
        ann["obj_to_world"].append(pose.tolist())
        ann["box_size"].append([4.6, 2.0, 1.7])

    with open(os.path.join(d, "instances", "instances_info.json"), "w") as fh:
        json.dump({"0": {"id": "abc", "class_name": "Vehicle", "frame_annotations": ann}}, fh)
    with open(os.path.join(d, "instances", "frame_instances.json"), "w") as fh:
        json.dump({str(f): [0] for f in range(N_FRAMES)}, fh)
    return d


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("waymo_ds"))
    _make_scene(root, "000", lidar_cols=14)
    _make_scene(root, "001", lidar_cols=10)
    _make_scene(root, "002", lidar_cols=None)  # no lidar at all
    # A scene whose lidar file is uniform noise, as some real scenes ship.
    d = _make_scene(root, "003", lidar_cols=None)
    os.makedirs(os.path.join(d, "lidar"), exist_ok=True)
    rng = np.random.default_rng(1)
    for f in range(N_FRAMES):
        (rng.uniform(-100, 100, 8000).astype(np.float32)).tofile(
            os.path.join(d, "lidar", f"{f:03d}.bin")
        )
    return root


def _ds(root, **kw):
    return WaymoDriveStudioClips(
        root=root,
        cameras=CAMS,
        num_context=4,
        num_target=2,
        frame_stride=5,
        clip_length=20,
        resolution=(IMG_H, IMG_W),
        deterministic=True,
        **kw,
    )


# ------------------------------------------------------------------- lidar


def _read_waymo_lidar(path):
    """Exercise the current instance-aware reader with automatic layout detection."""
    dataset = object.__new__(WaymoDriveStudioClips)
    dataset.lidar_columns = None
    return dataset.read_lidar(path)


def test_read_lidar_handles_both_column_layouts(dataset_root):
    for scene in ("000", "001"):
        p = os.path.join(dataset_root, scene, "lidar", "000.bin")
        pts = _read_waymo_lidar(p)
        assert pts is not None, f"{scene} should parse"
        assert pts.shape == (500, 3)
        # Points were written in the ego frame: x forward, y left, z up.
        assert pts[:, 0].min() >= 5.0 and pts[:, 0].max() <= 60.0
        assert abs(pts[:, 1]).max() <= 20.0


def test_read_lidar_rejects_placeholder_noise(dataset_root):
    p = os.path.join(dataset_root, "003", "lidar", "000.bin")
    assert _read_waymo_lidar(p) is None


def test_read_lidar_missing_file(tmp_path):
    assert _read_waymo_lidar(str(tmp_path / "nope.bin")) is None


def test_argoverse_lidar_reads_xyzi_layout(tmp_path):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    xyzi = np.column_stack((points, np.array([0.2, 0.8], dtype=np.float32)))
    path = tmp_path / "sweep.bin"
    xyzi.tofile(path)

    loaded = ArgoverseDriveStudioClips.read_lidar(str(path))

    assert loaded is not None
    assert loaded.dtype == np.float64
    assert np.array_equal(loaded, points)


def test_argoverse_crop_is_centered_on_calibrated_principal_point():
    crop_x, crop_y, crop_width, crop_height = ArgoverseDriveStudioClips._crop_parameters(
        original_width=1550,
        original_height=2048,
        target_width=768,
        target_height=576,
        principal_x=774.0,
        principal_y=1023.0,
    )

    assert (crop_width, crop_height) == (1550, 1162)
    assert crop_x == 0
    assert abs((1023.0 - crop_y) / crop_height - 0.5) < 1e-3


def test_require_lidar_filters_scenes(dataset_root):
    assert {s["name"] for s in _ds(dataset_root).scenes} == {"000", "001", "002", "003"}
    # 002 has no sweeps and 003 has only noise, so both drop out.
    assert {s["name"] for s in _ds(dataset_root, require_lidar=True).scenes} == {"000", "001"}


# -------------------------------------------------------- axis convention


def test_opencv2waymo_maps_forward_axis():
    """Ego +x (forward) must become camera +z; ego +z (up) becomes camera -y."""
    r = OPENCV2WAYMO[:3, :3]
    # The matrix is applied as cam_to_ego = extrinsics @ OPENCV2WAYMO, so its
    # columns are the OpenCV axes expressed in Waymo axes.
    assert np.allclose(r @ np.array([0, 0, 1.0]), [1, 0, 0])   # cam z -> ego x
    assert np.allclose(r @ np.array([1.0, 0, 0]), [0, -1, 0])  # cam x -> ego -y
    assert np.allclose(r @ np.array([0, 1.0, 0]), [0, 0, -1])  # cam y -> ego -z


def test_lidar_projects_in_front_of_camera(dataset_root):
    """With the convention applied, most of a forward sweep must land on-image.

    This is the check that caught the missing axis change: without it every
    point falls behind the camera and the sparse depth map comes back empty.
    """
    clip = _ds(dataset_root, scenes=["000"])[0]
    ctx = clip["context"]
    assert "sparse_depth" in ctx

    mask = ctx["sparse_mask"] > 0.5
    assert bool(mask.any()), "no lidar point projected into any view"

    depth = ctx["sparse_depth"][mask]
    assert float(depth.min()) > 0.0

    # Points were placed 5-60 m ahead in the EGO frame, and the camera sits
    # 1.5 m forward of the ego origin, so camera depths run 3.5-58.5 m.
    cam_offset = 1.5
    assert float(depth.min()) > 5.0 - cam_offset - 0.5
    assert float(depth.max()) < 60.0 - cam_offset + 0.5


def test_lidar_projection_is_deterministic_and_nearest_wins(tmp_path, monkeypatch):
    dataset = object.__new__(WaymoDriveStudioClips)
    points = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    monkeypatch.setattr(dataset, "read_lidar", lambda _: points)
    metadata = {"path": str(tmp_path), "extrinsics": {0: np.eye(4)}}
    intrinsics = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])

    first_depth, first_mask = dataset._lidar_depth(metadata, 0, 0, intrinsics, (5, 5))
    second_depth, second_mask = dataset._lidar_depth(metadata, 0, 0, intrinsics, (5, 5))

    assert torch.equal(first_depth, second_depth)
    assert torch.equal(first_mask, second_mask)
    assert float(first_depth[0, 2, 2]) == pytest.approx(2.0)
    assert int(first_mask.sum()) == 1


# ------------------------------------------------------------ scene graph


def test_boxes_are_world_space_and_self_consistent(dataset_root):
    clip = _ds(dataset_root, scenes=["000"])[0]
    boxes = clip["boxes"]
    assert boxes is not None and boxes.num_instances == 1

    for j in range(boxes.valid.shape[1]):
        if bool(boxes.valid[0, j]):
            assert bool(
                inside_box(boxes.centers[0, j], boxes.centers[0, j], boxes.dims[0, j], boxes.rots[0, j])
            )
    # box_size is (length, width, height) and maps to local (x, y, z).
    assert torch.allclose(boxes.dims[0, 0], torch.tensor([4.6, 2.0, 1.7]))


def test_world_is_recentred_on_first_frame(dataset_root):
    """Waymo world coords are ~1e4 m; float32 needs them brought near zero."""
    clip = _ds(dataset_root, scenes=["000"])[0]
    scene, targets, _ = to_scene_input(clip, torch.device("cpu"))

    cam_pos = scene.cam_to_world[:, :3, 3]
    assert float(cam_pos.abs().max()) < 500.0, "poses were not recentred"

    boxes = clip["boxes"]
    assert float(boxes.centers.abs().max()) < 500.0, "boxes were not recentred"

    # Recentring must be a rigid change of frame, so the camera-to-box distance
    # is unchanged from the raw annotation.
    raw_gap = 10_030.0 - 10_000.0  # instance ahead of ego at frame 0
    gap = float((boxes.centers[0, 0] - cam_pos[0]).norm())
    assert abs(gap - raw_gap) < 3.0


def test_recentring_preserves_relative_motion(dataset_root):
    """Recentring is one rigid transform for the whole clip, not per frame.

    So the instance's own 3 m/frame world motion must survive intact -- it must
    NOT be reduced to the 1 m/frame it moves relative to the ego vehicle.
    """
    clip = _ds(dataset_root, scenes=["000"])[0]
    boxes = clip["boxes"]
    frames = clip["frames"].tolist()

    first, last = 0, boxes.valid.shape[1] - 1
    travel = float((boxes.centers[0, last] - boxes.centers[0, first]).norm())
    assert travel == pytest.approx((frames[last] - frames[first]) * 3.0, abs=0.5)


def test_ego_motion_survives_recentring(dataset_root):
    """The ego's own 2 m/frame motion likewise shows up in the camera poses."""
    clip = _ds(dataset_root, scenes=["000"])[0]
    scene, _, _ = to_scene_input(clip, torch.device("cpu"))
    frames = clip["frames"].tolist()

    # View 0 and the last view of the context share camera 0 at different frames.
    order = scene.frame_index.tolist()
    i_first = order.index(min(order))
    i_last = len(order) - 1 - order[::-1].index(max(order))
    moved = float((scene.cam_to_world[i_last, :3, 3] - scene.cam_to_world[i_first, :3, 3]).norm())
    expected = (frames[max(order)] - frames[min(order)]) * 2.0
    assert moved == pytest.approx(expected, abs=1.0)


def test_clip_structure(dataset_root):
    ds = _ds(dataset_root, scenes=["000"])
    clip = ds[0]
    ctx, tgt = clip["context"], clip["target"]

    assert ctx["images"].shape[0] == 4 * len(CAMS)     # V = 4 frames x 3 cameras
    assert ctx["intrinsics"].shape == (4 * len(CAMS), 3, 3)
    assert ctx["cam_to_world"].shape == (4 * len(CAMS), 4, 4)
    assert tgt["images"].shape[0] == 2 * len(CAMS)

    # Timestamps are normalised to [0, 1] across the clip's frames.
    assert float(ctx["times"].min()) >= 0.0 and float(ctx["times"].max()) <= 1.0
    # frame_index indexes the box slots.
    assert int(ctx["frame_index"].max()) < clip["boxes"].valid.shape[1]


def test_intrinsics_rescaled_with_resize(dataset_root):
    """The stored K is for full-resolution images and must follow the resize."""
    ds = WaymoDriveStudioClips(
        root=dataset_root, scenes=["000"], cameras=CAMS, num_context=4, num_target=1,
        frame_stride=5, clip_length=20, resolution=(IMG_H // 2, IMG_W // 2),
        deterministic=True,
    )
    clip = ds[0]
    k = clip["context"]["intrinsics"][0]
    # Written as fx=100, cx=W/2 at (IMG_H, IMG_W); halving the size halves both.
    assert float(k[0, 0]) == pytest.approx(50.0, abs=1e-3)
    assert float(k[0, 2]) == pytest.approx(IMG_W / 4, abs=1e-3)


# ------------------------------------------------- loader -> model junction


def test_loader_output_drives_the_model(dataset_root):
    """The seam between the dataset and the model, which neither suite covers.

    test_smoke builds SceneInput by hand and test_data stops at the loader, so a
    mismatch in view count, dtype or box layout between the two would slip
    through both.  Everything up to rasterisation runs on CPU.
    """
    from pointforward.models import ConstantDepth, PointForward

    ds = _ds(dataset_root, scenes=["000"])
    clip = ds[0]
    scene, targets, gt = to_scene_input(clip, torch.device("cpu"))

    num_views = 4 * len(CAMS)
    assert scene.num_views == num_views
    assert scene.sparse_depth is not None and scene.sparse_mask is not None

    model = PointForward(
        depth_model=ConstantDepth(15.0),
        feat_dim=16,
        color_dim=16,
        hidden=16,
        num_static=200,
        num_dynamic=100,
        num_views=num_views,
        pretrained_backbone=False,
    )
    model.train()

    depth = model.predict_depth(scene)
    assert depth.shape == (num_views, IMG_H, IMG_W)

    sampled = model.build_queries(scene, depth)
    expected = num_views * 200 + clip["boxes"].num_instances * 100
    assert sampled["queries"].shape == (expected, 12)

    gaussians = model.decode(scene, depth, sampled)
    assert gaussians.count == expected
    assert torch.isfinite(gaussians.means).all()
    assert torch.isfinite(gaussians.colors).all()
    assert bool((gaussians.sigma >= 0).all())

    # Gradients must reach the trainable modules through the real data path.
    (gaussians.colors.square().mean() + gaussians.means.square().mean()).backward()
    for name in ("backbone", "fusion", "decoder"):
        grads = [
            p.grad for p in getattr(model, name).parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads and any(bool(g.abs().sum() > 0) for g in grads), name


def test_model_rejects_view_count_mismatch(dataset_root):
    """A config whose V disagrees with the loader must fail loudly, not silently."""
    from pointforward.models import ConstantDepth, PointForward

    clip = _ds(dataset_root, scenes=["000"])[0]
    scene, _, _ = to_scene_input(clip, torch.device("cpu"))

    model = PointForward(
        depth_model=ConstantDepth(15.0),
        feat_dim=16, color_dim=16, hidden=16,
        num_static=100, num_dynamic=50,
        num_views=scene.num_views + 3,     # wrong on purpose
        pretrained_backbone=False,
    )
    depth = model.predict_depth(scene)
    with pytest.raises(ValueError, match="num_views"):
        model.decode(scene, depth, model.build_queries(scene, depth))


def test_dynamic_queries_found_in_synthetic_scene(dataset_root):
    """The synthetic instance must actually capture some queries.

    ConstantDepth places every query on one depth plane, so this also pins the
    plane against the box the fixture puts 30 m ahead -- if either moved, the
    dynamic branch would go untested here and the test above would pass
    vacuously.
    """
    from pointforward.models import ConstantDepth, PointForward

    clip = _ds(dataset_root, scenes=["000"])[0]
    scene, _, _ = to_scene_input(clip, torch.device("cpu"))

    model = PointForward(
        depth_model=ConstantDepth(30.0),   # the instance sits ~30 m ahead
        feat_dim=16, color_dim=16, hidden=16,
        num_static=200, num_dynamic=100,
        num_views=scene.num_views,
        pretrained_backbone=False,
    )
    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    assert bool((sampled["instance_id"] >= 0).any()), "no query landed in the box"
