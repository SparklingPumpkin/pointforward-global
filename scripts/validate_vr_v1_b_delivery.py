"""Audit the frozen delivery and an already executed independent replay.

Does not run training or overwrite release evidence. Prints a JSON result;
the caller may use --out to save its own validation report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    release = project/'docs/releases/VR-v1-b'
    manifest = json.loads((release/'delivery_manifest.json').read_text())
    checked = []
    for row in manifest['files']:
        path = project/row['path']
        assert path.is_file() and not path.is_symlink(), path
        assert path.stat().st_size == row['bytes'], path
        assert sha256(path) == row['sha256'], path
        checked.append(row['path'])
    source_manifest = json.loads((release/'source_manifest.json').read_text())
    for row in source_manifest['files']:
        path = project/row['path']
        assert sha256(path) == row['delivered_sha256'], path
    original = json.loads((release/'evidence/original_render_manifest.json').read_text())
    replay = args.replay.resolve()
    actual = json.loads((replay/'runtime-render-source/manifest.json').read_text())
    keys = ['mean_psnr_source_frames_all_cameras',
            'mean_psnr_dynamic_bbox_source_frames_all_cameras',
            'mean_psnr_non_sky_source_frames_all_cameras']
    errors = {key: abs(actual[key]-original[key]) for key in keys}
    assert max(errors.values()) <= .0001, errors
    assert actual['frame_count'] == 156 and len(actual['cameras']) == 3
    probes = {}
    for name in ('final_global_runtime_predicted_front3.mp4', 'final_global_runtime_gt_vs_predicted_front3.mp4'):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'json', str(replay/name)], text=True))
        stream = probe['streams'][0]
        assert int(stream['nb_frames']) == 156 and stream['r_frame_rate'] == '10/1', probe
        probes[name] = stream
    preview = replay/'final_global_gaussians_frame000_preview.ply'
    assert preview.stat().st_size > 0
    expected_ply = next(r for r in manifest['files'] if r['path'].endswith('.ply'))
    assert sha256(preview) == expected_ply['sha256'], 'Independent PLY differs from frozen preview'
    generic = project/'weights/pointforward_waymo_ds_paper_ckpt_020000.pt'
    assert sha256(generic) == '318205d7475b4687dc59412f5a4432b5ae2a58da0ae143b5a69d1f18071d52ea'
    assert sha256(project/'weights/infinidepth_depthsensor.ckpt') == '53230af7c46e987acc993f250e76f99914e7321ee8a13966136fdb0619f98267'
    prior = json.loads((project/'docs/releases/FFR-v1-l-ver2/delivery_manifest.json').read_text())
    for row in prior['weights']+prior['artifacts']:
        path = project/row['path']
        assert path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], path
    result = dict(release='VR-v1-b', passed=True, checksum_files=len(checked),
        source_files=len(source_manifest['files']), file_checksums_verified=True,
        numerical_source_changes=source_manifest['existing_files_changed'],
        psnr={key:actual[key] for key in keys}, absolute_error=errors, videos=probes,
        ply_sha256_identical=True, scene='017', frame_count=156,
        previous_delivery_assets_verified=True,
        cold_start_retrained=False, replay_seconds=json.loads((replay/'pipeline_timing.json').read_text())['seconds'],
        note='Validates delivered state replay/export; does not claim newly measured cold-start training time.')
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
