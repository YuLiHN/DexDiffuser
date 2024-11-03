import os
import sys

sys.path.append(os.getcwd())

import pickle
from loguru import logger

import torch
import numpy as np

from models import create_ddpm, create_evaluator
import copy
import argparse
from bps_torch.bps import bps_torch
from utils.utils import load_ckpt
from utils.io import mkdir_if_not_exists
from tqdm import tqdm
from utils.handmodel import angle_denormalize, angle_normalize


class RefineNN:
    '''
    Do refinement with euler angle representation instead of qt representation

    '''
    def __init__(self, grasp_scoring_network, batch_max_size=20):
        self.grasp_scoring_network = grasp_scoring_network
        self.batch_max_size = batch_max_size
        self.pc = None
        self.device = torch.device('cuda:0')

    def improve_grasps_sampling_based(self,
                                      pc_with_grasp,
                                      num_refine_steps, 
                                      delta_translation=0.02):

        with torch.no_grad():
            grasp_seq = []
            score_seq = []
            grasps = pc_with_grasp['x_t']

            last_success = self.grasp_scoring_network(pc_with_grasp)['p_success']

            grasp_seq.append(grasps.cpu().numpy())
            score_seq.append(last_success.cpu().numpy())
            pc_with_newgrasp = copy.deepcopy(pc_with_grasp)

            for _ in range(num_refine_steps):
               
                delta_t = torch.rand(grasps.shape[0], 3).to(self.device)*delta_translation - delta_translation / 2.
                delta_r = torch.rand(grasps.shape[0], 6).to(self.device)*delta_translation - delta_translation / 2.
                delta_theta = torch.rand(grasps.shape[0], 16).to(self.device)*delta_translation - delta_translation / 2.

                delta = torch.hstack([delta_t, delta_r, delta_theta])
                perturbed_grasp = grasps + delta
                pc_with_newgrasp['x_t'] = perturbed_grasp
                perturbed_success = self.grasp_scoring_network(pc_with_newgrasp)['p_success']
                ratio = perturbed_success / torch.max(
                    last_success,
                    torch.tensor(0.0001).to(self.device))
                mask = torch.rand(ratio.shape).to(self.device) <= ratio

                ind = torch.where(mask)[0]
                last_success[ind] = perturbed_success[ind]
                grasps[ind] = perturbed_grasp.data[ind]

                grasp_seq.append(grasps.cpu().numpy())
                score_seq.append(last_success.cpu().numpy())

            grasp_seq = np.array(grasp_seq)
            score_seq = np.array(score_seq)

            ind_list = np.argmax(score_seq, axis=0)

            rows, cols = np.indices(ind_list.shape)
            last_success = score_seq[ind_list, rows, cols]
            grasps = grasp_seq[ind_list, rows].squeeze(axis=1)
            return grasps, last_success

    def improve_grasps_sampling_based_global(self,
                                      pc_with_grasp,
                                      num_refine_steps, delta_translation=0.02):

        with torch.no_grad():
            grasp_seq = []
            score_seq = []
            grasps = pc_with_grasp['x_t']

            last_success = self.grasp_scoring_network(pc_with_grasp)['p_success']

            grasp_seq.append(grasps.cpu().numpy())
            score_seq.append(last_success.cpu().numpy())
            pc_with_newgrasp = copy.deepcopy(pc_with_grasp)

            for _ in range(num_refine_steps):

                delta_t = torch.rand(grasps.shape[0], 3).to(self.device) * delta_translation - delta_translation / 2.

                delta_r = torch.rand(grasps.shape[0], 6).to(self.device) * delta_translation - delta_translation / 2.
                delta_theta = torch.zeros(grasps.shape[0], 16).to(self.device)

                delta = torch.hstack([delta_t, delta_r, delta_theta])
                perturbed_grasp = grasps + delta
                pc_with_newgrasp['x_t'] = perturbed_grasp
                perturbed_success = self.grasp_scoring_network(pc_with_newgrasp)['p_success']
                ratio = perturbed_success / torch.max(
                    last_success,
                    torch.tensor(0.0001).to(self.device))
                mask = torch.rand(ratio.shape).to(self.device) <= ratio

                ind = torch.where(mask)[0]
                last_success[ind] = perturbed_success[ind]
                grasps[ind] = perturbed_grasp.data[ind]

                grasp_seq.append(grasps.cpu().numpy())
                score_seq.append(last_success.cpu().numpy())

            grasp_seq = np.array(grasp_seq)
            score_seq = np.array(score_seq)

            ind_list = np.argmax(score_seq, axis=0)

            rows, cols = np.indices(ind_list.shape)
            last_success = score_seq[ind_list, rows, cols]
            grasps = grasp_seq[ind_list, rows].squeeze(axis=1)
            return grasps, last_success
    def improve_grasps_sampling_based_local(self,
                                      pc_with_grasp,
                                      num_refine_steps, delta_translation=0.02):

        with torch.no_grad():
            grasp_seq = []
            score_seq = []
            grasps = pc_with_grasp['x_t']

            last_success = self.grasp_scoring_network(pc_with_grasp)['p_success']

            grasp_seq.append(grasps.cpu().numpy())
            score_seq.append(last_success.cpu().numpy())
            pc_with_newgrasp = copy.deepcopy(pc_with_grasp)

            for _ in range(num_refine_steps):
                delta_t = torch.zeros(grasps.shape[0], 3).to(self.device)
                delta_r = torch.zeros(grasps.shape[0], 6).to(self.device)
                delta_theta = torch.rand(grasps.shape[0], 16).to(self.device) * delta_translation - delta_translation / 2.

                delta = torch.hstack([delta_t, delta_r, delta_theta])
                perturbed_grasp = grasps + delta
                pc_with_newgrasp['x_t'] = perturbed_grasp
                perturbed_success = self.grasp_scoring_network(pc_with_newgrasp)['p_success']
                ratio = perturbed_success / torch.max(
                    last_success,
                    torch.tensor(0.0001).to(self.device))
                mask = torch.rand(ratio.shape).to(self.device) <= ratio

                ind = torch.where(mask)[0]
                last_success[ind] = perturbed_success[ind]
                grasps[ind] = perturbed_grasp.data[ind]

                grasp_seq.append(grasps.cpu().numpy())
                score_seq.append(last_success.cpu().numpy())

            grasp_seq = np.array(grasp_seq)
            score_seq = np.array(score_seq)

            ind_list = np.argmax(score_seq, axis=0)

            rows, cols = np.indices(ind_list.shape)
            last_success = score_seq[ind_list, rows, cols]
            grasps = grasp_seq[ind_list, rows].squeeze(axis=1)
            return grasps, last_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test Scripts of Grasp Generation')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='evaluation directory path for saving results')
    parser.add_argument('--ckpt_evaluator', type=str, default='ckpts/10_4_0_evaluator/model_20.pth')
    parser.add_argument('--data_dir', type=str, default='/proj/berzelius-2023-338/users/x_haolu/dexdiffuser_data')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='run all on cuda')
    parser.add_argument('--two_stage_refinement', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, default='multidex')
    parser.add_argument('--num_refinement', type=int, default=100)
    parser.add_argument('--grasp_number_per_object', type=int, default=20)
    parser.add_argument('--delta_translation'   , type=float, default=0.001)

    return parser.parse_args()

def refine_grasp_dexgn(args):
    args = parse_args()

    evaluator = create_evaluator(pos_enc_multires=[10,4,4])
    device = 'cuda:0'
    evaluator.to(device=device)
    evaluator.eval()
    load_ckpt(evaluator, path=args.ckpt_evaluator)

    cam_number = 2
    grasp_number_per_object = args.grasp_number_per_object
    two_stage_refinement = args.two_stage_refinement
    num_refinement = args.num_refinement
    delta_translation = args.delta_translation

    grasp_ori = os.path.join(args.eval_dir, args.dataset_name, 'res_diffuser.pkl')
    # grasp_refinement_target = grasp_ori.split('.pkl')[0] + f'_refine_{num_refinement}_twostage{two_stage_refinement}.pkl'
    grasp_refinement_dir = os.path.join(args.eval_dir, args.dataset_name+f'twostage{two_stage_refinement}_{num_refinement}steps_delta_translation{delta_translation}')
    grasp_refinement_target = os.path.join(grasp_refinement_dir, 'res_diffuser.pkl')
    mkdir_if_not_exists(grasp_refinement_dir)

    refineNN = RefineNN(grasp_scoring_network=evaluator)
    obj_bps_dict = torch.load(os.path.join(args.data_dir,'obj_bps_dist_full.pt'))

    

    grasp_ori_pkl_all = pickle.load(open(grasp_ori, "rb"))
    grasp_ori_pkl = grasp_ori_pkl_all['sample_qpos']
    object_scale_list = ['0.06', '0.08', '0.1', '0.12', '0.15']
    for object_name in tqdm(grasp_ori_pkl.keys()):

        # object_name = "kit+FizzyTablets"
        grasp_grasp_ori = grasp_ori_pkl[object_name]
        for i,object_scale in enumerate(object_scale_list):

            for j in range(cam_number):
                # obj_bps_i = obj_bps_dict[object_name][pointcloud_id]
                # obj_pc_i = scene_pcds[object_name][pointcloud_id]
                # obj_bps_i = bps.encode(torch.from_numpy(obj_pc_i).unsqueeze(0).to(device), feature_type=['dists'])['dists']
                pointcloud_id = i*cam_number+j
                grasp_grasp_ori_i = grasp_grasp_ori[pointcloud_id*grasp_number_per_object:(pointcloud_id+1)*grasp_number_per_object]
                grasp_grasp_ori_i_torch = torch.Tensor(grasp_grasp_ori_i).contiguous().to(device)
                obj_bps_i = obj_bps_dict[object_name][object_scale][cam_number]
                obj_bps_i_torch = obj_bps_i.repeat([grasp_number_per_object, 1]).to(device)
                # data input preparation

                angle_norm = angle_normalize(joint_angle=grasp_grasp_ori_i_torch[:, 9:])
                grasp_grasp_ori_i_torch = torch.cat([grasp_grasp_ori_i_torch[:, :9], angle_norm], dim=1)

                data = {'x_t': grasp_grasp_ori_i_torch,
                        # 'scene_rot_mat': i_rot,
                        'obj_bps': obj_bps_i_torch}
            
                if not two_stage_refinement:
                    grasp_refine, output_succ = refineNN.improve_grasps_sampling_based(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
                else:
                    grasp_refine_global, output_succ = refineNN.improve_grasps_sampling_based_global(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
                    data['x_t'] = torch.tensor(grasp_refine_global).cuda()
                    grasp_refine, output_succ = refineNN.improve_grasps_sampling_based_local(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
                # obj_bps_dict[]

                grasp_refine = torch.from_numpy(grasp_refine).cuda()
                angle_denorm = angle_denormalize(joint_angle=grasp_refine[:, 9:])
                grasp_refine = torch.cat([grasp_refine[:, :9], angle_denorm], dim=1)
                grasp_refine = grasp_refine.cpu().numpy()

                grasp_ori_pkl_all['sample_qpos'][object_name][pointcloud_id*grasp_number_per_object:(pointcloud_id+1)*grasp_number_per_object] = \
                    grasp_refine

    with open(grasp_refinement_target, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(grasp_ori_pkl_all, f, pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Done! Refinement results are saved at {grasp_refinement_target}")


def refine_grasp_else(args):

    evaluator = create_evaluator(pos_enc_multires=[10,4,-1])
    device = 'cuda:0'
    evaluator.to(device=device)
    evaluator.eval()
    load_ckpt(evaluator, path=args.ckpt_evaluator)

    cam_number = 10
    grasp_number_per_object = args.grasp_number_per_object
    two_stage_refinement = args.two_stage_refinement
    num_refinement = args.num_refinement
    delta_translation = args.delta_translation

    grasp_ori = os.path.join(args.eval_dir, args.dataset_name, 'res_diffuser.pkl')
    # grasp_refinement_target = grasp_ori.split('.pkl')[0] + f'_refine_{num_refinement}_twostage{two_stage_refinement}.pkl'
    grasp_refinement_dir = os.path.join(args.eval_dir, args.dataset_name+f'_twostage{two_stage_refinement}_{num_refinement}steps_dt{delta_translation}')
    grasp_refinement_target = os.path.join(grasp_refinement_dir, 'res_diffuser.pkl')
    mkdir_if_not_exists(grasp_refinement_dir)

    refineNN = RefineNN(grasp_scoring_network=evaluator)
    scene_pcds = pickle.load(open(os.path.join(args.data_dir, f'pc_data_{args.dataset_name}.pickle'), 'rb'))['partial_pcs']
    basis_bps_set = np.load('./models/basis_point_set.npy')
    bps = bps_torch(n_bps_points=4096,
                    n_dims=3,
                    custom_basis=basis_bps_set)
    

    grasp_ori_pkl_all = pickle.load(open(grasp_ori, "rb"))
    grasp_ori_pkl = grasp_ori_pkl_all['sample_qpos']
    for object_name in grasp_ori_pkl.keys():

        # object_name = "kit+FizzyTablets"
        grasp_grasp_ori = grasp_ori_pkl[object_name]

        for pointcloud_id in range(cam_number):
            # obj_bps_i = obj_bps_dict[object_name][pointcloud_id]
            obj_pc_i = scene_pcds[object_name][pointcloud_id]
            obj_bps_i = bps.encode(torch.from_numpy(obj_pc_i).unsqueeze(0).to(device), feature_type=['dists'])['dists']

            grasp_grasp_ori_i = grasp_grasp_ori[pointcloud_id*grasp_number_per_object:(pointcloud_id+1)*grasp_number_per_object]
            grasp_grasp_ori_i_torch = torch.Tensor(grasp_grasp_ori_i).contiguous().to(device)
            obj_bps_i_torch = obj_bps_i.repeat([grasp_number_per_object, 1]).to(device)

            angle_norm = angle_normalize(joint_angle=grasp_grasp_ori_i_torch[:, 9:])
            grasp_grasp_ori_i_torch = torch.cat([grasp_grasp_ori_i_torch[:, :9], angle_norm], dim=1)
            
            # data input preparation
            data = {'x_t': grasp_grasp_ori_i_torch,
                    # 'scene_rot_mat': i_rot,
                    'obj_bps': obj_bps_i_torch}
        
            if not two_stage_refinement:
                grasp_refine, output_succ = refineNN.improve_grasps_sampling_based(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
            else:
                grasp_refine_global, output_succ = refineNN.improve_grasps_sampling_based_global(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
                data['x_t'] = torch.tensor(grasp_refine_global).cuda()
                grasp_refine, output_succ = refineNN.improve_grasps_sampling_based_local(data, num_refine_steps=num_refinement, delta_translation=delta_translation)
            # obj_bps_dict[]

            grasp_refine = torch.from_numpy(grasp_refine).cuda()
            angle_denorm = angle_denormalize(joint_angle=grasp_refine[:, 9:])
            grasp_refine = torch.cat([grasp_refine[:, :9], angle_denorm], dim=1)
            grasp_refine = grasp_refine.cpu().numpy()

            # print("output_succ", output_succ.mean())
            
            grasp_ori_pkl_all['sample_qpos'][object_name][pointcloud_id*grasp_number_per_object:(pointcloud_id+1)*grasp_number_per_object] = \
                grasp_refine

    with open(grasp_refinement_target, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(grasp_ori_pkl_all, f, pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Done! Refinement results are saved at {grasp_refinement_target}")

def main():
    args = parse_args()
    if args.dataset_name == 'dexgraspnet':
        refine_grasp_dexgn(args)
    else:
        refine_grasp_else(args)

if __name__ == "__main__":
    main()

