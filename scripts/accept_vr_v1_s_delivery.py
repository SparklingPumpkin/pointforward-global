"""Packaging-only workflow promotion, after independent replay and review."""
import json
from pathlib import Path
import subprocess


def main():
    project = Path(__file__).resolve().parents[1]
    base = project/'docs/releases/PFG-v3-a'
    prep = json.loads((base/'preparation.json').read_text())
    rr = prep['validation_dir']
    result = json.loads((project/rr/'validation.json').read_text())
    review = json.loads((project/rr/'pipeline_review.json').read_text())
    assert result['passed'] and review['passed'] and not result['cold_start_retrained']
    assert result['research_vehicle_29db_passed'] is False
    version = 'docs/releases/VR-v1-s/'
    evidence = {
        'assets': [version+'delivery_manifest.json::逐文件SHA256及原始来源',
            version+'source_manifest.json::全部交付源码指纹及明确的便携差异',
            rr+'/validation.json::3922项资产及443项源码验证通过',
            'artifacts/scene017_vr_v1_s/runtime-payload-refined/final_static_128d.pt::已冻结32D静态Gaussian',
            'artifacts/scene017_vr_v1_s/runtime-payload-refined/final_dynamic_128d.pt::已冻结32D动态Gaussian',
            'artifacts/scene017_vr_v1_s/runtime-payload-refined/global_dynamic_meta.pt::修正轨迹及固定评估框',
            'artifacts/scene017_vr_v1_s/runtime-payload-refined/feature_projection.pt::配套PCA投影基',
            'artifacts/scene017_vr_v1_s/pedestrian_tracks.pt::配套SMPL姿态',
            'weights/refiner_scene017_vr_v1_s.pt::配套32D U-Net',
            'weights/SMPL_NEUTRAL.npz::配套SMPL模板'],
        'pipeline': [rr+'/pipeline_review.json::fresh完整配方及9个真实入口import成功和原配置等价核对；未重训',
            version+'entrypoint_manifest.json::版本专属入口与模板指纹，默认别名允许维护',
            version+'evidence/config_build.original.yaml::原始冷启动数值配置',
            'scripts/run_vr_v1_s_pipeline.py::包内fresh/replay编排和完整真实计时',
            'scripts/validate_vr_v1_s_delivery.py::资产及数值重放校验实现',
            version+'source/scripts/delivery_runtime_helpers.py::可移植编排适配',
            version+'source/scripts/refine_unified_scene_rgb.py::原联合优化数值实现',
            version+'source/scripts/geometry_rgb_masks.py::包内SAM2参数适配',
            version+'source/sitecustomize.py::预编译扩展显式启用边界'],
        'replay': [rr+'/validation.json::完整156帧三相机三项PSNR误差零及PLY一致',
            rr+'/replay/runtime-render-source/manifest.json::原始独立重渲染指标',
            version+'evidence/original_render_manifest.json::源1670步原始指标',
            rr+'/replay/final_global_runtime_gt_vs_predicted_front3.mp4::独立完整GT对比视频',
            rr+'/replay/final_global_runtime_predicted_front3.mp4::独立完整预测视频',
            rr+'/replay/final_global_gaussians_frame000_preview.ply::独立导出且指纹一致的预览',
            rr+'/replay/review.png::亲自查看的20/80/140帧三相机对比',
            rr+'/REPORT.md::人工视觉检查与原有模糊缺陷'],
        'honesty': [version+'README.md::源冷启动阶段边界和完整交付耗时及本次仅固定状态重放',
            version+'evidence/pipeline_timing.json::源完整交付真实原始计时',
            version+'evidence/optimization_timing.json::源预处理到优化退出计时',
            version+'evidence/acceptance_report.json::研究车辆29dB失败未改写',
            version+'evidence/input_provenance.md::原始处理后输入边界和通用离线前提',
            rr+'/replay/pipeline_timing.json::本次仅cached_incremental重放计时',
            rr+'/REPORT.md::不宣称新冷启动训练或单PLY复现RGB指标']}
    contract = json.loads((base/'acceptance_contract.json').read_text())
    report = dict(schema_version=1, subject_id=prep['candidate'], checks=[
        dict(id=r['id'], criterion=r['criterion'], passed=True, run_id=prep['run_id'],
             evidence=evidence[r['id']]) for r in contract['criteria']])
    (base/'acceptance_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    workflow = '/mnt/fjn/FJN-Agent/skills/fjn-project-workflow/scripts/project_workflow.py'

    def wf(action, *args):
        result = subprocess.check_output(['python', workflow, action, '--project-root', str(project), *args], text=True)
        print(result)

    wf('set-run-status', '--run-id', prep['run_id'], '--status', 'completed',
       '--summary', '独立156帧三相机重放PSNR零误差、PLY一致、资产与源码审查通过；未重新冷启动训练',
       '--evidence', rr+'/validation.json::完整独立重放与指纹核验',
       '--evidence', rr+'/REPORT.md::人工审查及计时质量边界')
    wf('conclude-candidate', '--candidate-id', prep['candidate'], '--status', 'passed',
       '--summary', 'VR-v1-s第1670步32D运行时独立交付：全局31.131654、车辆28.329244dB精确复现；研究车辆29dB目标仍失败',
       '--evidence', rr+'/validation.json::完整独立重放及资产核验通过',
       '--acceptance-report', 'docs/releases/PFG-v3-a/acceptance_report.json')
    wf('promote', '--candidate-id', prep['candidate'], '--slug', 'compact32-union-smpl-delivery',
       '--summary', '当前交付VR-v1-s第1670步：车辆几何校正、32D Gaussian/PCA/U-Net、SMPL插值、全三相机union联合训练；fresh/replay便携入口；独立156帧全局31.131654/车辆28.329244dB，源完整RGB处理后输入至视频PLY1609.087s，研究29dB目标仍未达标',
       '--evidence', rr+'/validation.json::用户指定当前最佳交付，独立完整验证通过')


if __name__ == '__main__':
    main()
