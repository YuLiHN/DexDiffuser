import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any
from random import randint

from utils.handmodel import get_handmodel, angle_denormalize, trans_denormalize
from utils.plotly_utils import plot_mesh, plot_point_cloud
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot, identity_rot
from tqdm import tqdm
from bps_torch.bps import bps_torch
import numpy as np
import io
from PIL import Image
from loguru import logger

with open('dataset/test_split.txt', 'r') as file:
    # Read all lines from the file and store them in a list
    dexgraspnet_test = [line.strip() for line in file]

object_name_dict = \
    {
        'multidex':['contactdb+cube_medium', 'contactdb+cell_phone', 'contactdb+rubber_duck',
                    'contactdb+mouse', 'contactdb+piggy_bank', 'contactdb+pyramid_large',
                    'contactdb+camera', 'contactdb+binoculars', 'contactdb+pyramid_small',
                    'contactdb+cube_large', 'contactdb+pyramid_medium', 'contactdb+banana',
                    'contactdb+hammer', 'contactdb+ps_controller', 'contactdb+apple', 'contactdb+alarm_clock'],
        
        'egad':['egad+A0', 'egad+A1', 'egad+A2', 'egad+A3', 'egad+A4', 'egad+A5', 'egad+A6',
                'egad+B0', 'egad+B1', 'egad+B2', 'egad+B3', 'egad+B4', 'egad+B5', 'egad+B6',
                'egad+C0', 'egad+C1', 'egad+C2', 'egad+C3', 'egad+C4', 'egad+C5', 'egad+C6',
                'egad+D0', 'egad+D1', 'egad+D2', 'egad+D3', 'egad+D4', 'egad+D5', 'egad+D6',
                'egad+E0', 'egad+E1', 'egad+E2', 'egad+E3', 'egad+E4', 'egad+E5', 'egad+E6',
                'egad+F0', 'egad+F1', 'egad+F2', 'egad+F3', 'egad+F4', 'egad+F5', 'egad+F6',
                'egad+G0', 'egad+G1', 'egad+G2', 'egad+G3', 'egad+G4', 'egad+G5', 'egad+G6'],

        'dexgraspnet':dexgraspnet_test
    }
@torch.no_grad()
class GraspGenURVisualizer():
    def __init__(self, cfg:DictConfig) -> None:
        """ Visual evaluation class for pose generation task.
        Args:
            cfg: visuzalizer configuration
        """
        self.cfg = cfg
        self.ksample = cfg.task.visualizer.ksample
        self.hand_model = get_handmodel(batch_size=1, device='cuda', urdf_path=cfg.task.dataset.urdf_root, robot=cfg.task.dataset.robot_name)


        
        basis_bps_set = np.load('./models/basis_point_set.npy')
        self.bps = bps_torch(n_bps_points=4096,
                            n_dims=3,
                            custom_basis=basis_bps_set)
        
    def save_res(self, 
                       vis_type, 
                       scene_id, 
                       save_dir, 
                       outputs, 
                       obj_pcd_can, 
                       n_vis,
                       datasetname):
        if vis_type is not None:
            # scene_id = data['scene_id'][0]
            scene_dataset, scene_object = scene_id.split('+')
            if datasetname != 'dexgraspnet':
                mesh_dir = self.cfg.task.dataset.object_root
                mesh_path = os.path.join(mesh_dir, scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)
            else:
                pass

            for i in range(n_vis):
                self.hand_model.update_kinematics(q=outputs[i:i + 1, :])
                vis_data = [plot_mesh(obj_mesh, color='lightblue', opacity=0.5)]

                vis_data += self.hand_model.get_plotly_data(opacity=1.0, color='pink')
                if obj_pcd_can is not None:
                    vis_data += [plot_point_cloud(obj_pcd_can[i].cpu(), color='red')]
                save_path = os.path.join(save_dir, vis_type, f'{scene_id}+sample-{i}.{vis_type}')
                fig = go.Figure(data=vis_data)
                # fig.update_layout(
                                # scene = dict(
                                # xaxis = dict(visible=False),
                                # yaxis = dict(visible=False),
                                # zaxis =dict(visible=False)
                                # )
                                # )
                if vis_type == 'png':
                    fig.write_image(save_path)
                elif vis_type == 'html':
                    fig.write_html(save_path)
    
    @torch.no_grad()
    def sample_grasps(
            self,
            model: torch.nn.Module,
            datasetname: str, #['egad', 'condactdb', 'multidex']
            data_root: str,
            save_dir: str,
            cam_views: list = [0],
            guid_scale: float = None,
            evaluator: torch.nn.Module=None,
            device: str = 'cuda',
            num_sample:int =1,
            vis_type:str = 'html'
    ) -> None:
        """ Visualize method
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()        
        os.makedirs(os.path.join(save_dir, 'html'), exist_ok=True)

        if datasetname != 'dexgraspnet':
            scene_pcds = pickle.load(open(os.path.join(data_root, f'pc_data_{datasetname}.pickle'), 'rb'))
        else:
            obj_bps_all = torch.load(f"{data_root}/obj_bps_dist_full.pt")
        n_list = len(object_name_dict[datasetname])
        object_name_list = object_name_dict[datasetname]

        res = {'method': 'DexDiffuser',
               'desc': 'grasp generation using DexSampler',
               'sample_qpos': {},
               }
            
        if guid_scale is not None:
            guid_param = {'evaluator':evaluator,
                          'guid_scale': guid_scale}
            logger.info("using guided sampling")
        else:
            guid_param = None
            logger.info("using non-guided sampling")
        
        if datasetname=='dexgraspnet':
            scene_pcds = torch.load(f"{data_root}/scene_pcd_all.pt")

            assert abs(len(cam_views)%5)<1e-6 # divisiable by 5!
            num_grasp_per_scale = num_sample * len(cam_views)
            object_scale_grasp = [0.06] * num_grasp_per_scale + [0.08] * num_grasp_per_scale + [0.1] * num_grasp_per_scale + [0.12] * num_grasp_per_scale + [0.15] * num_grasp_per_scale
            res['scale_list'] = object_scale_grasp
            object_scale_list = ['0.06', '0.08', '0.1', '0.12', '0.15']
            p_success_list = []
            for object_name in tqdm(object_name_list):
                grasp_list = []
                for object_scale in object_scale_list:
                    # res['sample_qpos'][object_name] = {}
                    for cam_view in cam_views[:len(cam_views)//5]:
                        obj_bps = obj_bps_all[object_name][object_scale][cam_view].repeat(num_sample,1)
                        data = {'x': torch.randn(num_sample, self.cfg.model.d_x, device=device),
                                'obj_bps': obj_bps.to(device),
                                'scene_id': [object_name for i in range(num_sample)],
                        }
                        if True:
                            data['pos'] = scene_pcds[object_name][object_scale][cam_view].unsqueeze(0).repeat(num_sample,1, 1).to(device)
                    
                        outputs = model.sample(data, k=1,guid_param=guid_param).squeeze(1)[:, -1, :].to(torch.float32)
                        
                        ## denormalization
                        if self.cfg.task.dataset.normalize_x:
                            outputs[:,9:] = angle_denormalize(joint_angle=outputs[:,9:].to(torch.float32).cpu()).cuda()
                        if self.cfg.task.dataset.normalize_x_trans:
                            outputs[:, :3] = trans_denormalize(global_trans=outputs[:, :3].cpu()).cuda()

                        ## save visualization
                        if vis_type is not None:
                            self.save_res(vis_type, object_name, save_dir, outputs, None, num_sample, datasetname)
                        
                        if evaluator is not None:
                            data['x_t'] = outputs
                            p_success = evaluator(data)['p_success']
                            p_success_list.append(p_success.detach().cpu().numpy())
                        
                        grasp_list.append(outputs.cpu().detach().numpy())
                        # scale_list+= [object_scale for _ in range(num_sample)]
                        # import pdb;pdb.set_trace()

                # assert np.concatenate(grasp_list).shape[0] == 200
                res['sample_qpos'][object_name] = np.concatenate(grasp_list)
                # res['']
                    
            pickle.dump(res, open(os.path.join(save_dir, 'res_diffuser.pkl'), 'wb'))


        else:
            p_success_list = []
            for object_name in tqdm(object_name_list):
                grasp_list = []
                for cam_view in cam_views:
                    obj_pcd_can = torch.from_numpy(scene_pcds['partial_pcs'][object_name][cam_view]).unsqueeze(0).repeat(num_sample,1, 1)
                    
                    data = {'x': torch.randn(num_sample, self.cfg.model.d_x, device=device),
                            # 'pos': obj_pcd_rot.to(device),
                            'pos': obj_pcd_can.to(device),
                            # 'scene_rot_mat': i_rot,
                            'scene_id': [object_name for i in range(num_sample)],
                            'cam_trans': [None for i in range(num_sample)]}
                    data['obj_bps'] = self.bps.encode(obj_pcd_can,feature_type=['dists'])['dists']
                    outputs = model.sample(data, k=1,guid_param=guid_param).squeeze(1)[:, -1, :].to(torch.float32)
                    
                    ## denormalization
                    if self.cfg.task.dataset.normalize_x:
                        outputs[:,9:] = angle_denormalize(joint_angle=outputs[:,9:].to(torch.float32).cpu()).cuda()
                    if self.cfg.task.dataset.normalize_x_trans:
                        outputs[:, :3] = trans_denormalize(global_trans=outputs[:, :3].cpu()).cuda()

                    ## save visualization
                    if vis_type is not None:
                        self.save_res(vis_type, object_name, save_dir, outputs, obj_pcd_can, num_sample, datasetname)
                    
                    if evaluator is not None:
                        data['x_t'] = outputs
                        p_success = evaluator(data)['p_success']
                        p_success_list.append(p_success.detach().cpu().numpy())
                    
                    grasp_list.append(outputs.cpu().detach().numpy())

                res['sample_qpos'][object_name] = np.concatenate(grasp_list)
            pickle.dump(res, open(os.path.join(save_dir, 'res_diffuser.pkl'), 'wb'))
    
    
            
