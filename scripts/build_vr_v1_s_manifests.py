"""Packaging-only manifest generator. Run before delivery acceptance is frozen."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT/'docs/releases/VR-v1-s'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    prep = json.loads((PROJECT/'docs/releases/PFG-v3-a/preparation.json').read_text())
    origin = Path(prep['source_run'])/'source-snapshot'
    source = RELEASE/'source'
    allowed = {'sitecustomize.py': 'Explicit opt-in for pinned generic extension ABI',
        'scripts/refine_unified_scene_rgb.py': 'Portable render/save orchestration import only',
        'scripts/geometry_rgb_masks.py': 'SAM2 source and checkpoint path arguments only'}
    rows, changed = [], []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        rel = path.relative_to(source)
        delivered = sha(path)
        upstream = origin/rel
        if rel.parts[0] == 'SAM2':
            upstream = Path('/mnt/data/fjn/projects/tools/sam2')/Path(*rel.parts[1:])
        elif rel.parts[0] == 'InfiniDepth':
            upstream = Path('/mnt/data/why/InfiniDepth')/Path(*rel.parts[1:])
        old = sha(upstream) if upstream.is_file() else None
        if old is not None and delivered != old:
            assert str(rel) in allowed, f'Unexpected source change: {rel}'
            changed.append(str(rel))
        rows.append(dict(path=str(path.relative_to(PROJECT)), source=str(upstream) if old else None,
            source_sha256=old, delivered_sha256=delivered,
            portability_change=allowed.get(str(rel)) if delivered != old else None))
    assert set(changed) == set(allowed), changed
    save(RELEASE/'source_manifest.json', dict(release='VR-v1-s', files=rows,
        existing_files_changed=changed, numerical_equations_changed=False,
        new_helper='scripts/delivery_runtime_helpers.py'))
    copies = prep['copy_sources']
    weights = {
        'pointforward_waymo_ds_paper_ckpt_020000.pt': 'Generic PointForward initialization',
        'infinidepth_depthsensor.ckpt': 'Generic sparse-depth prior',
        'sam2.1_hiera_tiny.pt': 'Generic RGB segmentation',
        'pedestrian_head_ckpt_002000.pt': 'Generic canonical pedestrian head',
        'SMPL_NEUTRAL.npz': 'Generic neutral SMPL template',
        'refiner_scene017_vr_v1_s.pt': 'Scene017 step1670 rank32 U-Net; matching payload/projection required'}
    files = []
    paths = sorted((PROJECT/'artifacts/scene017_vr_v1_s').rglob('*'))
    paths += [PROJECT/'weights'/name for name in weights]
    for path in paths:
        if not path.is_file():
            continue
        assert not path.is_symlink(), path
        relative = str(path.relative_to(PROJECT))
        digest = sha(path)
        upstream = copies.get(relative)
        if upstream:
            assert sha(Path(upstream)) == digest, f'Asset differs: {relative}'
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest, source=upstream,
            purpose=weights.get(path.name, 'Frozen scene state, geometry input or visual evidence')))
    save(RELEASE/'delivery_manifest.json', dict(release='VR-v1-s', files=files,
        source_run=prep['source_run'], checkpoint_step=1670,
        metrics_apply_to='32D feature Gaussian + matching PCA/U-Net/trajectory/source compositor',
        ply_role='Frame000 vertex RGB preview only; not equivalent to runtime PSNR'))
    entrypoints = ['scripts/run_vr_v1_s_pipeline.py', 'scripts/validate_vr_v1_s_delivery.py',
        'docs/releases/VR-v1-s/config.template.yaml', 'docs/releases/VR-v1-s/pipeline_recipe.json',
        'docs/releases/VR-v1-s/README.md']
    save(RELEASE/'entrypoint_manifest.json', dict(files=[dict(path=p, sha256=sha(PROJECT/p)) for p in entrypoints],
        mutable_alias='scripts/run_pipeline.py intentionally excluded from immutable evidence'))
    manifest_path = PROJECT/'weights/manifest.json'
    wm = json.loads(manifest_path.read_text())
    current_names = {'weights/'+name for name in weights}
    wm['files'] = [r for r in wm['files'] if r['path'] not in current_names]
    wm['files'] += [r for r in files if r['path'] in current_names]
    wm['project_version'] = 'VR-v1-s'
    save(manifest_path, wm)
    print(json.dumps(dict(source_files=len(rows), asset_files=len(files), changed=changed)))


if __name__ == '__main__':
    main()
