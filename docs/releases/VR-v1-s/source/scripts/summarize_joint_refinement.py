"""Report joint Gaussian/refiner probes without replacing their raw evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', type=Path, required=True)
    a = p.parse_args()
    family = a.family.resolve()
    entries = []
    curves = []
    for candidate in ('FFR-v1-b', 'FFR-v1-c', 'FFR-v1-d'):
        for run in sorted((family/candidate).glob('run-*')):
            timer_path = run/'pipeline_timing.json'
            metrics_path = run/'runtime-render-source/manifest.json'
            if not timer_path.exists():
                continue
            timer = json.loads(timer_path.read_text())
            options = json.loads((run/'options.json').read_text())
            adaptation_path = run/'joint-adaptation/manifest.json'
            fit = json.loads(adaptation_path.read_text()) if adaptation_path.exists() else {}
            entry = dict(candidate=candidate, run_id=run.name, run=str(run),
                status=timer['status'], cold_seconds=timer.get('seconds'),
                group=options['joint_group'], feature_groups=options['joint_feature_groups'],
                requested_epochs=options['joint_epochs'], training_frame_stride=options['joint_frame_stride'],
                stages=[dict(name=x['name'], seconds=x.get('seconds'), status=x['status']) for x in timer['stages']],
                optimizer_steps=fit.get('steps'), stop_reason=fit.get('stop_reason'),
                epochs=[x for x in fit.get('records', []) if 'epoch_mean_mse' in x],
                diagnostics=fit.get('parameter_diagnostics'))
            if metrics_path.exists() and timer['status'] == 'completed':
                metrics = json.loads(metrics_path.read_text())
                assert metrics['frame_count'] == 156 and metrics['cameras'] == [0, 1, 2]
                assert metrics['refiner_state'] and 'joint-adaptation' in metrics['payload_dir']
                entry.update(global_psnr=metrics['mean_psnr_source_frames_all_cameras'],
                    non_sky_psnr=metrics['mean_psnr_non_sky_source_frames_all_cameras'],
                    dynamic_bbox_psnr=metrics['mean_psnr_dynamic_bbox_source_frames_all_cameras'],
                    eligible_1200=timer['seconds'] <= 1200,
                    gain_over_feedforward=metrics['mean_psnr_source_frames_all_cameras']-21.85491148630778,
                    gain_over_refiner_only=metrics['mean_psnr_source_frames_all_cameras']-23.174969428624863)
                montage = Image.new('RGB', (1536, 1236), 'white')
                draw = ImageDraw.Draw(montage)
                for i, frame in enumerate((0, 30, 60, 90, 120, 150)):
                    image = Image.open(run/f'runtime-render-source/frames/comparison/frame_{frame:04d}.jpg').resize((768, 384))
                    x, y = (i % 2)*768, (i // 2)*412
                    montage.paste(image, (x, y+28))
                    draw.text((x+8, y+6), f'{candidate} frame {frame}: GT above / predicted below', fill='black')
                montage.save(run/'keyframes_montage.jpg', quality=92)
            steps_path = run/'joint-adaptation/loss_steps.jsonl'
            rows = [json.loads(x) for x in steps_path.read_text().splitlines()] if steps_path.exists() else [x for x in fit.get('records', []) if 'step' in x]
            if rows:
                with (run/'loss_trend.csv').open('w') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['step', 'epoch', 'frame', 'rgb_mse', 'loss'])
                    writer.writeheader()
                    writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in rows)
                label = f'{candidate} g{options["joint_feature_groups"]} {run.name[:20]}'
                curves.append((label, rows))
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                x = [r['step'] for r in rows]
                y = np.array([r['rgb_mse'] for r in rows])
                axes[0].plot(x, y, alpha=.4, label='sampled training frames')
                if len(y) >= 10:
                    axes[0].plot(x[9:], np.convolve(y, np.ones(10)/10, mode='valid'), label='10-step mean')
                epochs = entry['epochs']
                axes[1].plot([r['epoch'] for r in epochs], [r['epoch_mean_mse'] for r in epochs], marker='o')
                axes[0].set(xlabel='Optimizer step', ylabel='RGB MSE', title=label)
                axes[0].legend()
                axes[1].set(xlabel='Epoch', ylabel='Training RGB MSE', title='Epoch average (not final evaluation PSNR)')
                for ax in axes: ax.grid(alpha=.25)
                fig.tight_layout(); fig.savefig(run/'loss_trend.png', dpi=160); plt.close(fig)
            entries.append(entry)

    valid = [x for x in entries if x.get('eligible_1200')]
    best = max(valid, key=lambda x:x['global_psnr']) if valid else None
    summary = dict(input_contract='Argoverse017 RGB+LiDAR+calibration+instances+masks and generic pretrained weights; no external scene state',
        measured_output='preview PLY plus final 128D/refiner source-view predicted and GT comparison videos',
        budget_seconds=1200, baseline_feedforward_psnr=21.85491148630778,
        baseline_refiner_only_psnr=23.174969428624863, entries=entries, best_eligible=best)
    (family/'joint_experiments_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 三组同步优化实验', '',
        '以 FG-v1-i 最佳前馈配置冷启动，不复用历史场景 payload/refiner。冷启动计时从所有模型子进程初始化之前开始，到最终 PLY 与两份视频写盘并 fsync 后结束。原始输入含 RGB、LiDAR、标定、ego pose、instance/bbox 和 masks；并非 RGB-only。通用预训练权重与编译缓存为离线前提。', '',
        '## 完整源视角结果', '',
        '| 候选 | 特征维度/组 | 轮次上限 | 冷启动 s | 全局 PSNR | 动态 bbox PSNR | ≤1200s |',
        '|---|---:|---:|---:|---:|---:|---|']
    for x in entries:
        f = lambda key: f'{x[key]:.6f}' if x.get(key) is not None else '—'
        lines.append(f'| {x["candidate"]}/{x["run_id"]} | {x["feature_groups"]} | {x["requested_epochs"]} | {f("cold_seconds")} | {f("global_psnr")} | {f("dynamic_bbox_psnr")} | {x.get("eligible_1200", "running")} |')
    lines += ['', '较早的 g4 预实验用四组增量覆盖 128D 通道；后续 g128 为逐点完整 128D 特征优化。组1为特征+U-Net；组2增加XYZ与尺度；组3增加旋转/opacity，g128版本还使时间中心和sigma可训练（scene持久化的超大sigma会令它们梯度接近零）。车辆ID、bbox轨迹、标定和构建时的preview/source RGB不参与优化。', '',
        '训练和交付使用相同的 0.75 static source-RGB 混合，组内每一步同时更新被选Gaussian参数和U-Net。上轮refiner-only是未混合输出上训练，因此与它的差异不应全归因于新增Gaussian变量。', '',
        '## Loss证据', '',
        '每个run内保存 `loss_trend.csv`、`loss_trend.png` 与 `joint-adaptation/manifest.json`；g128运行另外逐步保存 `joint-adaptation/loss_steps.jsonl`。训练MSE会随随机帧难度波动，epoch均值也反映整个学习过程，不能直接充当训练结束时全156帧的PSNR。正式排名只读完整渲染manifest。', '',
        '新运行在单个系统单调时钟内设置训练截止时间，为模型状态保存、PLY导出和视频预留时间；最终是否满足20min仍由实测timer判定。g4早期timer里的900秒预算字段沿用旧默认，不是本轮1200秒验收边界；这里按不改动原始计时的方式从end-start重新比较1200秒。', '',
        '## 最佳可交付结果', '']
    if best:
        run = Path(best['run'])
        lines += [f'{best["candidate"]} / {best["run_id"]}: **{best["global_psnr"]:.6f} dB**, **{best["cold_seconds"]:.3f} s**。', '',
            f'- [正式视频]({run}/runtime-render-source/final_global_runtime_predicted_front3.mp4)',
            f'- [GT对照]({run}/runtime-render-source/final_global_runtime_gt_vs_predicted_front3.mp4)',
            f'- [PLY预览]({run}/final_global_gaussians_frame000_preview.ply)',
            f'- [loss趋势]({run}/loss_trend.png)', '',
            'PSNR属于128D Gaussian特征+适配U-Net+源RGB混合的正式runtime渲染，不能由单独preview PLY复现；Gaussian几何变化也不代表NVS质量已经经过验证。']
    (family/'joint_experiments_report.md').write_text('\n'.join(lines)+'\n')
    if curves:
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, rows in curves:
            y = np.array([r['rgb_mse'] for r in rows])
            x = [r['step'] for r in rows]
            if len(y) >= 10:
                ax.plot(x[9:], np.convolve(y, np.ones(10)/10, mode='valid'), label=label)
            else: ax.plot(x, y, marker='o', label=label)
        ax.set(xlabel='Optimizer step', ylabel='Training RGB MSE (10-step mean for dense logs)', title='Joint Gaussian / UNet experiments; training loss is not final PSNR')
        ax.grid(alpha=.25); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(family/'joint_loss_comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(completed=sum(x['status']=='completed' for x in entries), total=len(entries), best=best), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
