"""Numerical and selection contracts for portable optional preprocessing."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from pointforward_preprocessing.depth_mask import repair
from pointforward_preprocessing.range import measure_scene, compress
from pointforward_preprocessing.sky import SkySettings, SkyView, build_sky, shell_directions


class ModularPreprocessing(unittest.TestCase):
    def test_repair_retains_near_objects_and_never_invents_sky(self):
        z = np.full((65, 65), 1000., dtype=np.float32)
        z[28:36, 30:34] = 20.
        sky = np.ones_like(z, dtype=bool)
        sky[48:] = False
        before = z.copy()
        repaired, removed = repair(z, sky)
        self.assertTrue(removed[30, 31])
        self.assertFalse(repaired[30, 31])
        self.assertTrue(repaired[1, 1])
        self.assertFalse((repaired & ~sky).any())
        np.testing.assert_array_equal(before, z)
        z[4, 4] = np.nan
        z[4, 5] = np.inf
        z[4, 6] = 0.
        z[4, 7] = -1.
        repaired, removed = repair(z, sky)
        self.assertFalse(removed[4, 4:8].any())

    def test_original_core_ast_definitions_unchanged(self):
        package = ROOT/'src/pointforward_preprocessing'
        actual = {}
        for path in package.glob('*.py'):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    actual[node.name] = node
        for source in json.loads((package/'provenance.json').read_text())['definitions']:
            node = actual[source['definition']]
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).replace(', type_params=[]', '').encode()).hexdigest()
            self.assertEqual(digest, source['definition_ast_sha256'], source['definition'])

    def test_measure_common_metric_frame_and_radial_boundary(self):
        measured = measure_scene([np.array([[-10., -20., -30.], [10., 20., 30.]])], [[0., 0., 0.]])
        self.assertEqual(measured['center'], [0., 0., 0.])
        self.assertAlmostEqual(measured['measured_radius_m'], np.sqrt(1400))
        self.assertAlmostEqual(measured['sky_radius_m'], 2*np.sqrt(1400))
        points = torch.tensor([[0., 0., 0.], [150., 0., 0.], [1000., 0., 0.]])
        changed = compress(points, torch.zeros(3), 150., 160., 1000.)
        torch.testing.assert_close(changed[:2], points[:2])
        self.assertEqual(changed[-1, 0], 160.)
        for args in [(2., 1.), (1., float('nan'))]:
            with self.assertRaises(ValueError):
                measure_scene([np.ones((1, 3))], [[0., 0., 0.]], *args)

    def test_finite_shell_obeys_actual_camera_ray(self):
        position = torch.tensor([[10., 5., 0.], [-4., -3., 6.]])
        rays = torch.tensor([[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]]).repeat(2, 1, 1, 1)
        geometry = dict(center=[0., 0., 0.], radius_m=45.)
        radial = shell_directions(position, rays, geometry)
        hits = radial*45.
        actual = torch.nn.functional.normalize(hits-position[:, None, None, :], dim=-1)
        torch.testing.assert_close(actual, rays, atol=2e-7, rtol=1e-6)

    def test_sky_complete_roundtrip_and_source_only_colors(self):
        h, w = 12, 16
        directions = torch.nn.functional.normalize(torch.stack(torch.meshgrid(
            torch.linspace(-.3, .3, h), torch.linspace(-.6, .6, w), indexing='ij')+ (torch.ones(h, w),), dim=-1), dim=-1)
        view = SkyView(1, 0, np.full((h, w, 3), .4, dtype=np.float32), np.ones((h, w), dtype=bool),
                       np.zeros((h, w), dtype=bool), torch.zeros(3), directions)
        sizing = dict(center=[0., 0., 0.], sky_radius_m=40., measured_diameter_m=40., sky_multiplier=2.)
        settings = SkySettings(longitude_cells=32, latitude_cells=16, boundary_erosion=1)
        state, records = build_sky(lambda: iter([view]), sizing, settings=settings, coverage=(16, 8))
        self.assertEqual(len(records), 1)
        self.assertGreater(state['metadata']['extrapolated_gaussian_count'], 0)
        self.assertEqual(int(state['observed'].sum()), state['metadata']['observed_gaussian_count'])
        torch.testing.assert_close(state['rgb_logits'].sigmoid(), torch.full_like(state['rgb_logits'], .4))
        torch.testing.assert_close(torch.linalg.vector_norm(state['means'], dim=-1), torch.full((len(state['means']),), 40.), atol=1e-5, rtol=1e-6)
        self.assertTrue(all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)))
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder)/'sky.pt'
            torch.save(state, f)
            got = torch.load(f, weights_only=True)
            torch.testing.assert_close(state['means'], got['means'], atol=0, rtol=0)

    def test_all_seven_module_combinations_and_no_implicit_dependencies(self):
        names = ['lim-r', 'depth-mask', 'scene-sky']
        for length in (1, 2, 3):
            for selected in itertools.combinations(names, length):
                command = [sys.executable, str(ROOT/'scripts/preprocess_scene.py'), '--config', '/does/not/need/to/exist',
                    '--out', '/tmp/pfg-preprocess-dry-only', '--modules', *selected, '--dry-run']
                plan = json.loads(subprocess.check_output(command, text=True))
                self.assertEqual(plan['modules'], list(selected))
                self.assertFalse(plan['optimization'])
                self.assertEqual(set(plan['outputs']), set(selected))
        command = [sys.executable, str(ROOT/'scripts/preprocess_scene.py'), '--config', '/unused', '--out', '/tmp/unused',
            '--modules', 'scene-sky', '--depth-dir', '/cache', '--dry-run']
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_no_runtime_research_imports(self):
        for path in (ROOT/'src/pointforward_preprocessing').glob('*.py'):
            text = path.read_text()
            self.assertNotIn('/mnt/', text, path.name)
            self.assertNotIn('feedforward-parameter-search', text, path.name)
            self.assertNotIn('replay_fpl', text, path.name)

    def test_absent_ego_masks_are_explicit_empty_masks(self):
        from pointforward_preprocessing.dataset import PointForwardViews
        view = PointForwardViews.__new__(PointForwardViews)
        view.dataset = SimpleNamespace(cameras=[0, 2], resolution=(3, 4),
            _collect=lambda *args, **kwargs: {'sky_mask': torch.ones(2, 1, 3, 4)})
        view.scene = {'has_ego_mask': False}
        view.slots, view.times, view.origin, view.mask_dir = {}, None, np.eye(4), None
        got = view.collect(1)
        self.assertEqual(tuple(got['ego_mask'].shape), (2, 1, 3, 4))
        self.assertEqual(int(got['ego_mask'].sum()), 0)

    def test_mask_consumer_preserves_frame_camera_order_without_mutating_class(self):
        from PIL import Image
        from pointforward_preprocessing.consumer import apply_masks
        class Dataset:
            cameras = [0, 2]
            def _collect(self, *args, **kwargs):
                return {'sky_mask': torch.ones(4, 1, 3, 4), 'other': 'preserved'}
        with tempfile.TemporaryDirectory() as folder:
            for frame in (1, 9):
                for cam in (0, 2):
                    Image.fromarray(np.full((3, 4), 255 if (frame, cam) == (9, 0) else 0, dtype='uint8')).save(
                        Path(folder)/f'{frame:03d}_{cam}.png')
            ds = apply_masks(Dataset(), folder)
            got = ds._collect({}, [1, 9], {}, None, (3, 4), np.eye(4), False)
            self.assertEqual(got['other'], 'preserved')
            self.assertEqual(got['sky_mask'].sum(dim=(1, 2, 3)).tolist(), [0., 0., 12., 0.])
            self.assertEqual(int(Dataset()._collect()['sky_mask'].sum()), 48)


if __name__ == '__main__':
    unittest.main()
