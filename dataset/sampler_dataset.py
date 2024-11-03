from typing import Any, Tuple, Dict
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pickle as pkl
from utils.handmodel import angle_normalize, trans_normalize

from omegaconf import DictConfig, OmegaConf

class DexGraspNetSamplerAllegro(Dataset):
    def __init__(self, mode: str, cfg:DictConfig, robot_name='allegro') -> None:
        super(DexGraspNetSamplerAllegro, self).__init__()
        self.mode = mode
        # read split
        self.split = []
        with open(f'dataset/{self.mode}_split.txt', 'r') as f:
            for line in f:
                self.split.append(line.strip())

        self.dataset_name = cfg.task.dataset.name
        # self.is_downsample = cfg.is_downsample
        # self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.task.dataset.num_points
        self.use_color = cfg.task.dataset.use_color
        self.use_normal = cfg.task.dataset.use_normal
        self.normalize_x = cfg.task.dataset.normalize_x
        self.normalize_x_trans = cfg.task.dataset.normalize_x_trans
        self.obj_dim = 3
        self.num_partial = 10
        self.use_obj_bps = cfg.model.scene_model.name=='obj_bps'
        self.obj_scale_list = [0.06, 0.08, 0.1, 0.12, 0.15]

        ## resource folders
        self.data_root = cfg.task.dataset.data_root
        self.data_dir = self.data_root
        self.object_dir = cfg.task.dataset.object_root

        ## load data
        self._pre_load_data()
    
    def _pre_load_data(self) -> None:
        """ 
        Load dataset
        """
        self.frames = []
        self.scene_pcds = {}

        grasp_dataset = torch.load(os.path.join(self.data_dir, 'train_succ.pt'))

        
        if self.use_obj_bps:
            self.object_bps = torch.load(os.path.join(self.object_dir,'obj_bps_dist_full.pt'))
        else:
            self.scene_pcds = torch.load(os.path.join(self.object_dir,'scene_pcd_all.pt'))
        
        for i,grasp in tqdm(enumerate(grasp_dataset['grasp_data']), total=len(grasp_dataset['grasp_data'])):
            joint_angle = grasp.clone().detach()[9:] # 9 for the root, 16 for allegro params
            global_trans = grasp.clone().detach()[:3]
            rot_6d = grasp.clone().detach()[3:9]

            if self.normalize_x:
                joint_angle = angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = trans_normalize(global_trans)

            grasp = torch.cat([global_trans,rot_6d, joint_angle], dim=0).requires_grad_(True)
                
            self.frames.append({
                                'object_name': grasp_dataset['object_code'][i],
                                'grasp': grasp,
                                'object_scale': grasp_dataset['object_scales'][i]}
                                )
        print(f'Finishing Pre-load in DexGraspNetAllegro, {len(self.frames)} grasp in total')
    
    def __len__(self):
        return len(self.frames) * self.num_partial


    def __getitem__(self, index: Any):

        frame_index = index // self.num_partial
        pc_index = index % self.num_partial
        frame = self.frames[frame_index]
        scene_id = frame['object_name']

        if self.mode != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed

        grasp_qpos = (
            frame['grasp']
        )

        data = {
            'x': grasp_qpos,
            'scene_id': scene_id,
            'object_scale': frame['object_scale'],
        }

        ## load data, containing scene point cloud and point pose
        
        if self.use_obj_bps:
            obj_scale = frame['object_scale'].item()
            rounded_obj_scale = "{:.2f}".format(obj_scale)
            if rounded_obj_scale == '0.10':
                rounded_obj_scale = '0.1'
            obj_bps = self.object_bps[scene_id][rounded_obj_scale][pc_index]
            data['obj_bps'] = obj_bps.squeeze().cpu()
        else:
            obj_scale = frame['object_scale'].item()
            rounded_obj_scale = "{:.2f}".format(obj_scale)
            if rounded_obj_scale == '0.10':
                rounded_obj_scale = '0.1'
            scene_pc = self.scene_pcds[scene_id][rounded_obj_scale][pc_index]
            data['pos'] = scene_pc[:,:3]

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)