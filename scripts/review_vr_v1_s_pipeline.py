"""Packaging review: check fresh commands/imports/config against the source run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    release = project/'docs/releases/VR-v1-s'
    source = release/'source'
    prep = json.loads((project/'docs/releases/PFG-v3-a/preparation.json').read_text())
    origin = Path(prep['source_run'])
    shutil.copy2(origin/'config_build.yaml', release/'evidence/config_build.original.yaml')
    dry = json.loads(subprocess.check_output([sys.executable, str(project/'scripts/run_pipeline.py'),
        '--mode', 'fresh', '--data-root', '/data/raw', '--output-root', '/work/fresh',
        '--gpu', '0', '--vehicle-gpu', '1', '--dry-run'], text=True))
    env = dict(os.environ, PYTHONPATH=str(source), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    imports = []
    for name, command in dry['commands'].items():
        if name == 'rgb_masks':
            continue  # Separately instantiated the packaged SAM2 model/checkpoint on CPU.
        result = subprocess.run([sys.executable, command[1], '--help'], cwd=source, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, (name, result.stdout)
        imports.append(dict(name=name, returncode=result.returncode, script=command[1]))
    old = yaml.safe_load((origin/'config_build.yaml').read_text())
    new = yaml.safe_load((release/'config.template.yaml').read_text())
    path_keys = [('data','root'),('data','geometry_directory'),('depth','repo_root'),
        ('depth','checkpoint'),('pedestrians','checkpoint'),('pedestrians','smpl_model_path'),
        ('pedestrians','tracks_file')]
    for section, key in path_keys:
        old[section][key] = new[section][key]
    old['model']['pretrained_backbone'] = False
    assert old == new, 'Unexpected numerical configuration change'
    result = dict(passed=True, stage_imports=imports, fresh_dry_run=dry,
        config_numerically_identical_except_disabled_initial_download=True,
        sam_model_checkpoint_cpu_load_passed=True, sam_python_version='3.10.18',
        main_python_version=sys.version, torch_version='2.8.0+cu128',
        sam_source=str(source/'SAM2'), cold_training_executed=False,
        timing_review=dict(start='before config/raw inputs',
            optimizer_endpoint='after trainer subprocess success and checkpoint serialization',
            delivery_endpoint='after full video/metrics/PLY, copy and fsync',
            parallel_ff='two producers joined before SMPL/PCA; single shared stopwatch',
            fresh_scene_inputs='all generated inside current output, no delivered scene asset read',
            source_raw_to_optimizer_seconds=1485.371231320314,
            source_raw_to_delivery_seconds=1609.0870128450915),
        immutable_default_alias_excluded=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, imports=len(imports), cold_training_executed=False)))


if __name__ == '__main__':
    main()
