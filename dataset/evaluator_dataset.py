import os
import pickle
# import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.handmodel import angle_normalize, trans_normalize

from tqdm import tqdm
class DexGraspNetEvaluatorDataset(Dataset):
    
    def __init__(self, cfg:DictConfig, mode:str = 'train') -> None:
        super(DexGraspNetEvaluatorDataset, self).__init__()

        self.mode = mode
        self.split = []
        with open(f'dataset/{mode}_split.txt', 'r') as f:
            for line in f:
                self.split.append(line.strip())
        self.data_root = cfg.task.dataset.data_root
        self.frames = []
        self.num_partial = 10 # TODO: FIX FOR DIFFERENT VIEWPOINT LATER
        self.normalize_x = cfg.task.dataset.normalize_x
        self.normalize_x_trans = cfg.task.dataset.normalize_x_trans

        self._pre_load_data()
        
    def _pre_load_data(self):
        
        grasp_dataset = torch.load(os.path.join(self.data_root, f'regenerate/eva_{self.mode}.pt'))
        self.object_bps = torch.load(os.path.join(self.data_root,'obj_bps_dist_full.pt'))

        label_tensor=torch.tensor(grasp_dataset['label'])
        print(f"in total: {label_tensor.shape[0]}, succ: {label_tensor.sum()}" )

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
                                'object_scale': grasp_dataset['object_scales'][i],
                                'label': torch.tensor(grasp_dataset['label'][i]).float()}
                                )
        print(f'Finishing Pre-load in DexGraspNetAllegro, {len(self.frames)} grasp in total')

    def __len__(self):

        return len(self.frames) * self.num_partial
    
    def __getitem__(self, index):

        frame_index = index // self.num_partial
        pc_index = index % self.num_partial
        frame = self.frames[frame_index]
        scene_id = frame['object_name']

        data = {
                'label': frame['label'],
                'obj_name':frame['object_name'],
                'x_t': frame['grasp']
                }
        
        obj_scale = frame['object_scale'].item()
        rounded_obj_scale = "{:.2f}".format(obj_scale)
        if rounded_obj_scale == '0.10':
            rounded_obj_scale = '0.1'
        obj_bps = self.object_bps[scene_id][rounded_obj_scale][pc_index]
        data['obj_bps'] = obj_bps.squeeze().cpu()

        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)