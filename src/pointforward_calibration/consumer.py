"""Opt-in bridge for frozen reconstruction code; legacy geometry stays intact."""
from pathlib import Path
import json
import numpy as np
import torch


def install():
    from pointforward.data.vehicle_geometry import GeometryArgoverseClips
    from pointforward.data.drivestudio import WaymoDriveStudioClips
    if getattr(GeometryArgoverseClips,'_lim_r_bridge',False):return
    original_enable=GeometryArgoverseClips.enable_geometry
    original_camera=GeometryArgoverseClips._cam_to_world
    original_depth=WaymoDriveStudioClips._lidar_depth

    def enable(self,directory):
        original_enable(self,directory)
        path=Path(directory)/'application_manifest.json'
        if not path.exists():return  # historical adaptive delivery
        manifest=json.loads(path.read_text())
        if manifest.get('algorithm')!='LIM-v1-r':return
        if manifest['resolution']!=list(self.resolution):raise ValueError('LIM-r resolution mismatch')
        if not manifest['frames']:raise ValueError('Empty calibrated frame set')
        if not set(self.cameras)<=set(manifest['cameras']):raise ValueError('LIM-r camera set mismatch')
        cameras=torch.load(Path(directory)/'camera_poses.pt',map_location='cpu',weights_only=False)
        for meta in self.scenes:
            if meta['name']!=manifest['scene']:raise ValueError('LIM-r scene identity mismatch')
            meta['frames']=[f for f in meta['frames'] if f in set(manifest['frames'])]
            meta['_lim_r_camera_poses']=cameras
        span=max((self.num_context-1)*self.frame_stride,self.clip_length-1)
        self.index=[(si,start) for si,m in enumerate(self.scenes)
                    for start in range(0,len(m['frames'])-span)]
        self.lim_r_manifest=manifest

    def camera(self,meta,frame,cam,world_from):
        if '_lim_r_camera_poses' not in meta:return original_camera(self,meta,frame,cam,world_from)
        return world_from@np.asarray(meta['_lim_r_camera_poses'][f'{frame:03d}_{cam}'])

    def depth(self,meta,frame,cam,k,resolution):
        if not hasattr(self,'geometry_directory'):
            return original_depth(self,meta,frame,cam,k,resolution)
        data=np.load(self.geometry_directory/'sparse'/f'{frame:03d}_{cam}.npz')
        if tuple(data['depth'].shape)!=tuple(resolution):raise ValueError('LIM-r sparse depth resolution mismatch')
        value=torch.from_numpy(data['depth'].copy()).unsqueeze(0)
        return value,(value>0).float()

    GeometryArgoverseClips.enable_geometry=enable
    GeometryArgoverseClips._cam_to_world=camera
    GeometryArgoverseClips._lim_r_bridge=True
    WaymoDriveStudioClips.enable_geometry=enable
    WaymoDriveStudioClips._cam_to_world=camera
    WaymoDriveStudioClips._lidar_depth=depth
    # The frozen Waymo builder predates geometry_directory; Argoverse already
    # enables it. Opt in only for new calibrated data through this wrapper.
    from scripts import train
    build=train.build_dataset
    def build_dataset(cfg,split):
        dataset=build(cfg,split)
        directory=cfg['data'].get('geometry_directory')
        if directory and not hasattr(dataset,'geometry_directory'):
            dataset.enable_geometry(directory)
        return dataset
    train.build_dataset=build_dataset
