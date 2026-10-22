"""Validate frozen VR-v1-s asset identities and an independent full-scene replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    release = project/'docs/releases/VR-v1-s'
    manifest = json.loads((release/'delivery_manifest.json').read_text())
    for row in manifest['files']:
        path = project/row['path']
        assert path.is_file() and not path.is_symlink(), path
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
    source = json.loads((release/'source_manifest.json').read_text())
    for row in source['files']:
        path = project/row['path']
        assert not path.is_symlink() and sha(path) == row['delivered_sha256'], path
    entries = json.loads((release/'entrypoint_manifest.json').read_text())
    for row in entries['files']:
        assert sha(project/row['path']) == row['sha256'], row['path']
    actual = json.loads((args.replay/'runtime-render-source/manifest.json').read_text())
    expected = json.loads((release/'evidence/original_render_manifest.json').read_text())
    keys = ['mean_psnr_source_frames_all_cameras',
            'mean_psnr_dynamic_bbox_source_frames_all_cameras',
            'mean_psnr_non_sky_source_frames_all_cameras']
    errors = {key: abs(actual[key]-expected[key]) for key in keys}
    assert max(errors.values()) <= .0001, errors
    assert actual['frame_count'] == 156 and actual['cameras'] == [0, 1, 2]
    assert actual['feature_dim'] == 32 and actual['status'] == 'completed'
    probes = {}
    for name in ('final_global_runtime_predicted_front3.mp4', 'final_global_runtime_gt_vs_predicted_front3.mp4'):
        data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', str(args.replay/name)], text=True))
        stream = data['streams'][0]
        assert int(stream['nb_frames']) == 156 and stream['r_frame_rate'] == '10/1', stream
        probes[name] = stream
    ply = args.replay/'final_global_gaussians_frame000_preview.ply'
    frozen_ply = next(r for r in manifest['files'] if r['path'].endswith('.ply'))
    assert sha(ply) == frozen_ply['sha256'], 'PLY differs from original'
    assert sha(project/'weights/pointforward_waymo_ds_paper_ckpt_020000.pt') == '318205d7475b4687dc59412f5a4432b5ae2a58da0ae143b5a69d1f18071d52ea'
    assert sha(project/'weights/infinidepth_depthsensor.ckpt') == '53230af7c46e987acc993f250e76f99914e7321ee8a13966136fdb0619f98267'
    timer = json.loads((args.replay/'pipeline_timing.json').read_text())
    assert timer['scope'] == 'cached_incremental' and timer['status'] == 'completed'
    result = dict(release='VR-v1-s', passed=True, checksum_files=len(manifest['files']),
        source_files=len(source['files']), portability_changes=source['existing_files_changed'],
        numerical_equations_changed=False, psnr={k: actual[k] for k in keys}, absolute_error=errors,
        videos=probes, ply_sha256_identical=True, frame_count=156, feature_dim=32,
        cold_start_retrained=False, replay_seconds=timer['total_delivery_seconds'],
        research_vehicle_29db_passed=False,
        note='Independent package replay/export verified. Source cold timing is historical evidence, not this replay.')
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
