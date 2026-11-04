"""Meaningful contracts for the accepted calibration and its data bridge."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/releases/VR-v1-s/source'))
sys.path.insert(1,str(ROOT/'src'))
import numpy as np
from pointforward_calibration import c,r
from pointforward_calibration.pipeline import source_frames,zbuffer


class CalibrationContract(unittest.TestCase):
    def test_uniform_real_sources_exclude_both_ends(self):
        for offset,expected in [(1,[0,1,2]),(-1,[1,2,3])]:
            kept,excluded=source_frames([0,1,2,3],offset,offset,lambda f:True)
            self.assertEqual(kept,expected)
            self.assertEqual(len(excluded),1)
        kept,_=source_frames([0,1,2,3],1,-1,lambda f:f!=2)
        self.assertEqual(kept,[2])

    def test_continuous_independent_clocks_and_bounded_support(self):
        mats=np.repeat(np.eye(4)[None],3,axis=0);mats[:,0,3]=[0.,1.,2.]
        adapter=SimpleNamespace(actors={7:c.Poses([0.,.1,.2],mats)},
            base=SimpleNamespace(transform=lambda p,t:p@t[:3,:3].T+t[:3,3]))
        row=dict(track=7,lidar_time=.1,bbox_time=0.,camera_time=.15,
                 pool=np.array([[1.,0.,0.]]),selected_box=np.eye(4))
        # 1ms is a true intermediate trajectory time, not an integer frame.
        np.testing.assert_allclose(r.warp(adapter,row,.001),[[1.49,0.,0.]],atol=1e-10)
        box=r.warp(adapter,row,.02,box=True)
        self.assertAlmostEqual(box[0,3],1.3)
        self.assertIsNone(r.pose(adapter,7,1.)[0])

    def test_zbuffer_keeps_nearest_depth_and_does_not_move_input(self):
        points=np.array([[0.,0.,5.],[0.,0.,2.],[0.,0.,-1.]])
        before=points.copy();k=np.array([[1.,0.,1.],[0.,1.,1.],[0.,0.,1.]])
        depth=zbuffer(points,np.eye(4),k,(3,3))
        self.assertEqual(depth[1,1],2.)
        self.assertEqual(np.count_nonzero(depth),1)
        np.testing.assert_array_equal(points,before)

    def test_frozen_numerical_definitions(self):
        package=ROOT/'src/pointforward_calibration'
        manifest=json.loads((package/'numerical_contract.json').read_text())
        checked=0
        for file,definitions in manifest['definitions'].items():
            tree=ast.parse((package/file).read_text())
            actual={}
            for node in tree.body:
                if isinstance(node,ast.FunctionDef):actual[node.name]=node
                elif isinstance(node,ast.ClassDef):
                    for method in node.body:
                        if isinstance(method,ast.FunctionDef):actual[node.name+'.'+method.name]=method
            for name,expected in definitions.items():
                self.assertEqual(hashlib.sha256(ast.dump(actual[name],include_attributes=False).replace(", type_params=[]", "").encode()).hexdigest(),expected,file+':'+name)
                checked+=1
        self.assertGreater(checked,45)
        for name,sha in manifest['configs'].items():
            self.assertEqual(hashlib.sha256((package/name).read_bytes()).hexdigest(),sha)

    def plan(self,*args):
        cmd=[sys.executable,str(ROOT/'scripts/run_pipeline.py'),'--data-root','/data',
             '--output-root','/tmp/calibration-contract','--dry-run',*args]
        return json.loads(subprocess.check_output(cmd,text=True))

    def test_default_fresh_and_legacy_replay(self):
        fresh=self.plan('--mode','fresh')
        self.assertEqual(fresh['vehicle_calibration']['algorithm'],'LIM-v1-r')
        self.assertIn('lim_r_preprocessing',fresh['commands'])
        self.assertNotIn('rgb_masks',fresh['commands'])
        self.assertIn('run_calibrated_stage.py',' '.join(fresh['commands']['static_build']))
        legacy=self.plan('--mode','fresh','--vehicle-calibration','adaptive')
        self.assertIn('rgb_masks',legacy['commands'])
        self.assertNotIn('run_calibrated_stage.py',' '.join(legacy['commands']['static_build']))
        replay=self.plan('--mode','replay')
        self.assertEqual(replay['vehicle_calibration']['mode'],'adaptive')
        self.assertTrue(replay['scene_state_reused'])


if __name__=='__main__':unittest.main()
