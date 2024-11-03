import os
from requests import get
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from random import shuffle, randint
import yaml
from envs.base_task import BaseTask
from tqdm import tqdm
# import ipdb
from scipy.spatial.transform import Rotation as R
import copy
import numpy as np
import math
import functools
from typing import Optional
from envs.tasks.utils.angle import r2euler
import torch
import transforms3d
from utils.handmodel import get_handmodel
import torch.nn.functional as F
import trimesh as tm
import time
from math import pi
import pdb



class IsaacGraspTestForce_allegro(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 init_opt_q: torch.tensor, object_name: str, object_scales=None,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
                 fix_object=False, robot='allegro', mesh_path='mesh_path'):
        self.gym = None
        self.viewer = None
        self.num_sim = 0
        self.Kp = 1.0
        self.Kd = 0.1
        self.cfg = cfg
        self.sim_params = sim_params
        self.device = device_type
        self.init_opt_q = init_opt_q.clone()
        self.init_q = self.q_transfer_o2s(init_opt_q)
        self.object_name = object_name
        # self.object_volume = object_volume
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.fix_object = fix_object
        self.device = "cuda"
        self.robot = robot #['allegro','allegro_right']
        self.mesh_path = mesh_path

        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]

        cfg["env"]["numTrain"] = init_opt_q.shape[0]
        self.env_num_train = cfg["env"]["numTrain"]
        self.env_num = self.env_num_train
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        self.num_train = cfg["env"]["asset"]["AssetNumTrain"]
        self.tot_num = self.num_train
        train_list_len = len(cfg["env"]["asset"]["trainObjAssets"])
        self.train_name_list = []
        self.exp_name = cfg['env']["env_name"]

        print("Simulator: number of objects", self.tot_num)
        print("Simulator: number of environments", self.env_num)

        if object_scales is None:
            object_scales = [1. for _ in range(self.tot_num)]
        else:
            self.object_scales = object_scales

        if self.num_train:
            assert (self.env_num_train % self.num_train == 0)

        # the number of used length must less than real length
        assert (self.num_train <= train_list_len)
        # the number of used length must less than real length
        # each object should have equal number envs
        assert (self.env_num % self.tot_num == 0)
        self.env_per_object = self.env_num // self.tot_num

        for name in cfg["env"]["asset"]["trainObjAssets"]:
            self.train_name_list.append(
                cfg["env"]["asset"]["trainObjAssets"][name]["name"])

        self.env_ptr_list = []
        self.obj_loaded = False
        self.dexterous_loaded = False
        self.hand_idxs = []

        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg["env"]["enableCameraSensors"],
                         cam_pos=(0.5, 0., 0.0), cam_target=(0, 0, 0.0))

        # acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)).to(self.device)
        self.dof_state_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)).to(self.device)
        self.rigid_body_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)).to(self.device).reshape(self.num_envs, -1, 13)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(
            self.num_envs, -1, 2)
        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        self.initial_obj_state = self.initial_root_states[:, 1, :3]

        self.dexterous_dof_tensor = self.dof_state_tensor[:, :self.dexterous_num_dofs, :]

        self.dexterous_root_tensor = self.root_tensor[:, 0, :]
        self.object_root_tensor = self.root_tensor[:, 1, :]

        self.dof_dim = self.dexterous_num_dofs
        self.pos_act = self.initial_dof_states[:, :self.dexterous_num_dofs, 0].clone()
        self.eff_act = torch.zeros(
            (self.num_envs, self.dof_dim), device=self.device)

        self.error_pd = torch.zeros((self.num_envs, 16)).to(self.device)
        
        # Real
        # joint_name = self.gym.get_actor_joint_names(env_ptr, dexterous_actor)

        # ['move_x', 'move_y', 'move_z',
        #   'rot_r', 'rot_p', 'rot_y',
        #     'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip', 
        #     'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_15.0_tip', 
        #     'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_7.0_tip', 
        #     'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_11.0_tip']

        # Puhao:
        # WRJ2 WRJ1
        # FFJ4 FFJ3 FFJ2 FFJ1
        # MFJ4 MFJ3 MFJ2 MFJ1
        # RFJ4 RFJ3 RFJ2 RFJ1
        # LFJ5 LFJ4 LFJ3 LFJ2 LFJ1
        # THJ5 THJ4 THJ3 THJ2 THJ1
        # rigid props

        # ik parameters
        self.damping = 0.05
        self.prepare_ik()

    def __del__(self):
        if self.gym is not None:
            self.gym.destroy_sim(self.sim)
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        del self.gym


    def _get_num_envs(self):
        return max(self.len_per_object.values())

    def is_obj_stable(self, start_pos, end_pos):
        """
        :param start_pos: [B x 3]
        :param end_pos: [B x 3]
        :return:
        """
        is_obj_stable = torch.ones(start_pos.shape[0]).bool().to(self.device)
        error_distance = self.cfg['eval_policy']['error']['distance']

        obj_distance = torch.norm(start_pos - end_pos, dim=1)
        is_obj_stable *= (abs(obj_distance) < error_distance)
        return is_obj_stable

    def step_sim_q(self, sim_q):

        '''
        args: sim_q: [N, 22], [N, :3] for position, [N, 3:6] for rotation (euler angle), [N, 6:22] for allegro joint angles
        '''

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)


    def _set_init_pose(self):

        for i_iter in range(self.cfg['eval_policy']['init']['steps']):
            self.step_sim_q(self.init_q.clone())

    def _set_normal_force_pose(self):
        step_size = 0.2 # 0.2
        learning_rate = 0.5 # .5
        contact_threshold = 0.005
        num_object_pts = 2048
        object_pts = self.object_mesh.sample(num_object_pts)
        object_pts = torch.tensor(object_pts, device=self.device)

        # opt_q_global = self.init_opt_q[:, :11].clone().detach().to(self.device)
        # opt_q_angle = self.init_opt_q[:, 11:].clone().detach().to(self.device)
        opt_q_global = self.init_opt_q[:, :9].clone().detach().to(self.device)
        opt_q_angle = self.init_opt_q[:, 9:].clone().detach().to(self.device)
        # opt_q_global.requires_grad = True
        opt_q_angle.requires_grad = True
        opt_q = torch.cat([opt_q_global, opt_q_angle], dim=1)
        optimizer = torch.optim.Adam([opt_q_global, opt_q_angle], lr=learning_rate)
        hand_model = get_handmodel(robot=self.robot, batch_size=self.env_num, device=self.device, hand_scale=1.)
        surface_points, surface_normal = hand_model.get_surface_points_and_normals(q=opt_q)
        surface_points = surface_points.reshape(-1, 3)
        surface_normal = surface_normal.reshape(-1, 3)
        with torch.no_grad():
            surface_points_distance2mesh = surface_points.clone().reshape(-1, 1, 3).repeat(1, num_object_pts, 1)

            surface_points_distance2mesh -= object_pts

            surface_points_distance2mesh = surface_points_distance2mesh.norm(dim=2)
            surface_points_distance2mesh_values = torch.min(surface_points_distance2mesh, dim=1).values
            normal_mask = torch.zeros_like(surface_points_distance2mesh_values, device=self.device)
            normal_mask[surface_points_distance2mesh_values < contact_threshold] = 1.
            normal_mask = normal_mask.reshape(-1, 1).repeat(1, 3)

        surface_points_target = (surface_points.clone() + step_size * normal_mask * surface_normal.clone()).detach()
        loss = (surface_points - surface_points_target).norm(dim=1).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.closure_q = torch.cat([opt_q_global.clone().detach(), opt_q_angle.clone().detach()], dim=1)
        self.closure_q.requires_grad = False
        
        # self.closure_q = self.init_opt_q + torch.clamp(self.closure_q - self.init_opt_q, min=-0.3, max=0.3)
        # self.closure_q = self.q_transfer_o2s(self.closure_q)
        self.closure_q = self.q_transfer_o2s(opt_q)

        for i_iter in range(self.cfg['eval_policy']['init']['steps']):
            self.step_sim_q(self.closure_q.clone())

    def push_object(self):
        # import pdb;pdb.set_trace()
        self.init_q = self.init_q.contiguous()
        self._set_init_pose()
        # self._set_normal_force_pose()
        achieve_6dir = torch.ones(self.num_envs).bool().to(self.device)
        self.force_num_steps = self.cfg['eval_policy']['dynamic']['num_steps']
        self.object_force_directions = self.cfg['eval_policy']['dynamic']['directions'].values()
        with tqdm(range(self.force_num_steps * 6), desc='Test Push Object:') as pbar:
            # time.sleep(0.1)
            for i_direction in self.object_force_directions:
                i_achieve = self._push_object_with_direction(i_direction, pbar)
                achieve_6dir *= i_achieve
                
        self.error_pd = torch.zeros((self.num_envs, 16)).to(self.device)

        return achieve_6dir

    def _push_object_with_direction(self, i_direction, pbar):
        # object_force_magnitude = self.cfg['eval_policy']['dynamic']['magnitude_per_volume'] * self.object_volume 
        object_force_magnitude = self.cfg['eval_policy']['dynamic']['magnitude_per_volume'] * torch.tensor(self.object_volume_list, device='cuda').reshape(-1,1)
        object_pos_start = self.get_obj_pos()
        for i_iter in range(self.force_num_steps):

            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)


            force_position = self.rigid_body_tensor[:, :, :3].clone()
            object_force = torch.zeros_like(force_position, device='cuda')
            object_force[:, -1, :] = object_force_magnitude * torch.tensor(i_direction, device='cuda').unsqueeze(0)

            self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(object_force),
                                                           gymtorch.unwrap_tensor(force_position), gymapi.ENV_SPACE)



            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            # # todo: for debug
            if not self.headless:
                self.render()
            if self.cfg["env"]["enableCameraSensors"] is True:
                self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            pbar.update()
        object_pos_terminal = self.get_obj_pos()
        return self.is_obj_stable(object_pos_start, object_pos_terminal)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(
            self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._place_agents(
            self.cfg["env"]["numTrain"], self.cfg["env"]["envSpacing"])

    def _load_agent(self, env_ptr, env_id):

        if self.dexterous_loaded == False:
            self.dexterous_actor_list = []
            asset_root = self.asset_root
            # dexterous_asset_file = "movable_hand_urdf/allegro/robots/movable_allegro.urdf"
            if self.robot == 'allegro':
                dexterous_asset_file = "movable_hand_urdf/allegro/robots/allegro_hand_description_left.urdf"
            elif self.robot == 'allegro_right':
                dexterous_asset_file = "movable_hand_urdf/allegro/robots/allegro_hand_description_right.urdf"
            else:
                raise NotImplementedError
            asset_options = gymapi.AssetOptions()
            asset_options.density = self.cfg['agent']['density']
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = False
            asset_options.armature = 0.01
            asset_options.use_mesh_materials = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.override_com = True  # recompute center of mesh
            asset_options.override_inertia = True  # recompute inertia
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            # asset_options.vhacd_params.resolution = 3000000
            asset_options.vhacd_params.resolution = 3000000

            self.dexterous_asset = self.gym.load_asset(
                self.sim, asset_root, dexterous_asset_file, asset_options)
            self.dexterous_loaded = True

        dexterous_dof_max_torque, self.dexterous_dof_lower_limits, self.dexterous_dof_upper_limits = self._get_dof_property(
            self.dexterous_asset)

        dof_props = self.gym.get_asset_dof_properties(self.dexterous_asset)
        if self.cfg["env"]["driveMode"] in ["pos", "ik"]:
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:].fill(self.cfg['agent']['dof_props']['stiffness'])
            dof_props["velocity"][:].fill(self.cfg['agent']['dof_props']['velocity'])
            dof_props["damping"][:].fill(self.cfg['agent']['dof_props']['damping'])
        else:  # osc
            dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:].fill(0.0)
            dof_props["damping"][:].fill(0.0)

        # root pose
        initial_dexterous_pose = gymapi.Transform()
        # initial_dexterous_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # initial_dexterous_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        
        # convert from root 
        q = convert_euler_to_qt(self.init_q[env_id, 3:6].cpu().numpy())[0]
        tr = self.init_q[env_id, :3].cpu().numpy()
        initial_dexterous_pose.r = gymapi.Quat(q[0], q[1], q[2], q[3])
        initial_dexterous_pose.p = gymapi.Vec3(tr[0], tr[1], tr[2])

        # set start dof
        self.dexterous_num_dofs = self.gym.get_asset_dof_count(self.dexterous_asset)



        dexterous_dof_state = np.zeros_like(
            dexterous_dof_max_torque, gymapi.DofState.dtype)
        
        default_dof_pos = self.init_q[env_id, 6:].cpu().numpy() + 0.2 # grasp pose
        # default_dof_pos[5] -= 0.3

        dexterous_dof_state["pos"] = default_dof_pos

        dexterous_actor = self.gym.create_actor(
            env_ptr,
            self.dexterous_asset,
            initial_dexterous_pose,
            "dexterous",
            env_id,
            0,
            0)

        dexterous_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, dexterous_actor)
        for shape in dexterous_shape_props:
            shape.friction = self.cfg['agent']['shape']['friction']

        self.gym.set_actor_rigid_shape_properties(env_ptr, dexterous_actor, dexterous_shape_props)
        self.gym.set_actor_dof_properties(env_ptr, dexterous_actor, dof_props)
        self.gym.set_actor_dof_states(
            env_ptr, dexterous_actor, dexterous_dof_state, gymapi.STATE_ALL)
        
        self.gym.set_actor_dof_position_targets(env_ptr, dexterous_actor, 
                                                    default_dof_pos)
        
        props = self.gym.get_actor_dof_properties(env_ptr, dexterous_actor)
        props['driveMode'].fill(gymapi.DOF_MODE_POS)
        props['stiffness'].fill(1e10)  # A large value to lock the DOFs
        props['damping'].fill(1e10)  # A large value to lock the DOFs
        props['effort'].fill(10)  
        self.gym.set_actor_dof_properties(env_ptr, dexterous_actor, props)

        self.dexterous_actor_list.append(dexterous_actor)

        self.dexterous_link_dict = self.gym.get_asset_rigid_body_dict(self.dexterous_asset)


        joint_name = self.gym.get_actor_joint_names(env_ptr, dexterous_actor)


    def prepare_ik(self):

        # get dof state tensor
        self.dof_pos = self.dexterous_dof_tensor[:, :, 0].view(self.num_envs, self.dexterous_num_dofs, 1)
        self.dof_vel = self.dexterous_dof_tensor[:, :, 1].view(self.num_envs, self.dexterous_num_dofs, 1)

        # Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.effort_action = torch.zeros_like(self.pos_action)

    def q_transfer_o2s(self, opt_q):
        """
        :param opt_q:
        :return:
        """
        opt_q = opt_q.detach().clone()
        opt_translation = opt_q[:, :3].float().clone()
        sim_translation = opt_translation.clone()
        opt_rotation = opt_q[:, 3:9].clone()
        opt_joint_angle = opt_q[:, 9:].clone()

        rot_matrix = robust_compute_rotation_matrix_from_ortho6d(opt_rotation).cpu()
        sim_rpy = torch.zeros(rot_matrix.shape[0], 3, device=self.device)
        for i in range(rot_matrix.shape[0]):
            # print('------**------')
            # print(rot_matrix[i])
            sim_rpy[i] = torch.tensor(r2euler(rot_matrix[i], type="XYZ"), device=self.device)
            # # TODO: for debug
            # print(get_rot6d_from_rpy(sim_rpy[i]))
        # quit()
        sim_joint_angle = torch.zeros_like(opt_joint_angle, device=self.device)

        # hand model joint ['joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
        # 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
        # 'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
        # 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0']

        # isaac gym joint ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip', 
        # 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_15.0_tip', 
        # 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_7.0_tip', 
        # 'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_11.0_tip']

        # ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip', 
        #  'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_15.0_tip', 
        #  'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_7.0_tip', 
        #  'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_11.0_tip']

        sim_joint_angle[:, :4] = opt_joint_angle[:, 8:12]
        sim_joint_angle[:, 4:8] = opt_joint_angle[:, 12:16]
        sim_joint_angle[:, 8:12] = opt_joint_angle[:, 4:8]
        sim_joint_angle[:, 12:16] = opt_joint_angle[:, 0:4]

        # print("sim rpy: ", sim_rpy)
        # print("sim_translation: ", sim_translation)

        # sim_translation = torch.zeros_like(sim_translation)
        # sim_rpy = torch.zeros_like(sim_rpy)
        pos_action = torch.cat([sim_translation, sim_rpy, sim_joint_angle], dim=1)
        # pos_action = torch.cat([sim_translation, opt_rotation, sim_joint_angle], dim=1)
        return pos_action

    def step_opt_q(self, opt_q):
        # opt_q = opt_q.detach().cpu()
        opt_q = opt_q.to(self.device)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.pos_action = self.q_transfer_o2s(opt_q)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.num_sim += 1
        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def get_q_as_opt(self):
        # print(self.dof_state_tensor[:, :, 0].size())
        # refresh tensors
        sim_q_state = self.dof_state_tensor[:, :, 0].clone()
        sim_translation = sim_q_state[:, :3]
        sim_rpy = sim_q_state[:, 3:6]
        sim_q_angle = sim_q_state[:, 6:]

        opt_translation = sim_translation
        opt_rotation = torch.zeros(sim_q_state.shape[0], 6, device=self.device)
        for i in range(sim_q_state.shape[0]):
            opt_rotation[i, :] = torch.tensor(get_rot6d_from_rpy(sim_rpy[i, :]), device=self.device)
        opt_q_angle = torch.zeros_like(sim_q_angle, device=self.device)

        # opt_q_angle[:, -4:] = sim_q_angle[:, :4] 
        # opt_q_angle[:, 8:12] = sim_q_angle[:, 4:8]
        # opt_q_angle[:, :4] = sim_q_angle[:, 8:12]
        # opt_q_angle[:, 4:8] = sim_q_angle[:, -4:]

        opt_q_angle[:, 8:12] = sim_q_angle[:, :4]
        opt_q_angle[:, 12:16] = sim_q_angle[:, 4:8]
        opt_q_angle[:, 4:8] = sim_q_angle[:, 8:12] 
        opt_q_angle[:, 0:4] = sim_q_angle[:, 12:16]


        # opt_q_angle = sim_q_angle
        return torch.cat([opt_translation, opt_rotation, opt_q_angle], dim=1)

    def get_obj_pos(self):
        return self.rigid_body_tensor[:, -1, :3].detach().clone()

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num):
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _load_obj_asset(self):

        self.obj_name_list = []
        self.obj_asset_list = []
        self.table_asset_list = []
        self.obj_pose_list = []
        self.table_pose_list = []
        self.obj_actor_list = []
        self.table_actor_list = []

        train_len = len(self.cfg["env"]["asset"]["trainObjAssets"].items())
        train_len = min(train_len, self.num_train)
        total_len = train_len
        used_len = min(total_len, self.tot_num)

        select_train_asset = [i for i in range(train_len)]

        with tqdm(total=used_len) as pbar:
            pbar.set_description('Loading assets:')
            cur = 0

            obj_asset_list = []
            # prepare the assets to be used
            if self.object_name is None:
                raise NotImplementedError

            else:

                
                object_asset_options = gymapi.AssetOptions()
                object_asset_options.density = self.cfg['object']['density'] # * 0.1
                object_asset_options.linear_damping = self.cfg['object']['damping']['linear']
                object_asset_options.angular_damping = self.cfg['object']['damping']['angular']
                object_asset_options.override_com = True
                object_asset_options.override_inertia = True
                object_asset_options.fix_base_link = self.fix_object
                object_asset_options.use_mesh_materials = True
                object_asset_options.disable_gravity = True


                # dataset_name = self.object_name.split('+')[0]
                # object_name = self.object_name.split('+')[1]
                # object_urdf_path = f'data/object/{dataset_name}/{object_name}/{object_name}.urdf'
                # # object_mesh_path = f'assets/object/{dataset_name}/{object_name}/{object_name}.stl'
                # object_mesh_path = f'data/object/{dataset_name}/{object_name}/{object_name}.stl'
                # self.object_mesh = tm.load(object_mesh_path)
                if ('egad' in self.object_name) or ('contactdb' in self.object_name):
                    dataset_name = self.object_name.split('+')[0]
                    object_name = self.object_name.split('+')[1]
                    
                    object_mesh_path = f'./data/object/{dataset_name}/{object_name}/{object_name}.stl'
                    self.object_mesh = tm.load(object_mesh_path)
                    self.object_volume = self.object_mesh.volume
                    self.object_volume_list= [self.object_volume for _ in range(self.num_envs)]

                    object_urdf_path = f'data/object/{dataset_name}/{object_name}/{object_name}.urdf'
                    obj_asset = self.gym.load_asset(
                        self.sim, './', object_urdf_path, object_asset_options)
                
                else: 
                    # object_mesh_path = tm.load(os.path.join(data_root_path, self.object_name, "coacd", "decomposed.obj"), force="mesh", process=False)
                    object_urdf_path = os.path.join(self.mesh_path, self.object_name, "coacd")
                    self.object_mesh = tm.load(os.path.join(self.mesh_path, self.object_name, "coacd", "decomposed.obj"), force="mesh", process=False)
                    # self.object_mesh = tm.load(object_mesh_path)
                    # self.object_mesh = self.object_mesh.apply_scale(0.1)
                    self.object_volume = self.object_mesh.volume 
                    self.object_volume_list = []

                    for scale in self.object_scales:
                        object_mesh_clone = copy.copy(self.object_mesh)
                        object_mesh_scale = object_mesh_clone.apply_scale(scale)
                        self.object_volume_list.append(object_mesh_scale.volume * 0.6)
    
                    obj_asset = self.gym.load_asset(
                        self.sim, object_urdf_path, 'coacd.urdf', object_asset_options)
                
                # print(f"object: {self.object_name} volume: {self.object_volume}")
                    
                self.obj_asset_list.append(obj_asset)
                rig_dict = self.gym.get_asset_rigid_body_dict(obj_asset)
                self.obj_rig_name = list(rig_dict.keys())[0]
                obj_start_pose = gymapi.Transform()
                obj_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
                obj_start_pose.r = gymapi.Quat(0., 0., 0., 1.)
                self.obj_pose_list.append(obj_start_pose)

    def _load_obj(self, env_ptr, env_id):

        if self.obj_loaded == False:
            self._load_obj_asset()
            self.obj_loaded = True

        obj_type = env_id // self.env_per_object
        subenv_id = env_id % self.env_per_object
        obj_actor = self.gym.create_actor(
            env_ptr,
            self.obj_asset_list[obj_type],
            self.obj_pose_list[obj_type],
            "obj{}-{}".format(obj_type, subenv_id),
            env_id,
            0,
            0)
        # print("env_id: ", env_id)
        # print("load scale: ", self.object_scales[env_id])
        self.gym.set_actor_scale(env_ptr, obj_actor, self.object_scales[env_id])

        obj_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        for shape in obj_shape_props:
            shape.friction = self.cfg['object']['shape']['friction']
        self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_shape_props)

        self.obj_actor_list.append(obj_actor)
        # assert(self.gym.get_actor_rigid_body_names(self.env_ptr_list[0], 1)[0] == 'object')

    def _place_agents(self, env_num, spacing):

        print("Simulator: creating agents")

        lower = gymapi.Vec3(-spacing, -spacing, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing / 2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing / 2
        num_per_row = int(np.sqrt(env_num))

        with tqdm(total=env_num) as pbar:
            pbar.set_description('Enumerating envs:')
            for env_id in range(env_num):
                env_ptr = self.gym.create_env(
                    self.sim, lower, upper, num_per_row)
                # self.error_pd = torch.zeros((self.num_envs, 16)).to(self.device)
                self.env_ptr_list.append(env_ptr)
                self._load_agent(env_ptr, env_id)
                self._load_obj(env_ptr, env_id)
                pbar.update(1)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 1.
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _draw_line(self, src, dst):
        line_vec = np.stack([src, dst]).flatten().astype(np.float32)
        color = np.array([1, 0, 0], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(
            self.viewer,
            self.env_ptr_list[0],
            self.env_num,
            line_vec,
            color
        )


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda)
         @ dpose).view(num_envs, -1)
    return u


def relative_pose(src, dst):
    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)


def get_sim_param():
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1. / 60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 0
    return sim_params


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


# def get_rot6d_from_rot3d(rot3d):
#     global_rotation = np.array(transforms3d.euler.euler2mat(rot3d[0], rot3d[1], rot3d[2], axes='sxyz'))
#     return global_rotation.T.reshape(9)[:6]
def get_rot6d_from_rpy(rpy):
    rpy_inverse = - rpy
    global_rotation = np.array(
        transforms3d.euler.euler2mat(rpy_inverse[0], rpy_inverse[1], rpy_inverse[2], axes='sxyz'))
    return global_rotation.reshape(9)[:6]


def convert_euler_to_qt(euler):

    """get qt from euler
    Args:
        
        euler: matrices, (B, 3)

    Returns:
        qt: bacth arr, (B, 4) # wxyz
       
    """

    if len(euler.shape) == 1:
        euler = np.expand_dims(euler, 0)

    r = R.from_euler("XYZ", euler)
    q = r.as_quat() # xyzw

    return np.asarray(q)
    

def convert_transformation_matrix_to_qt(transformation_matrix):
    """get T from qt
    Args:
        
        transformation_matrix: matrices, (B, 4, 4)

    Returns:
        qt: bacth arr, (B, 7) # wxyz, xyz
       
    """
    if len(transformation_matrix.shape) == 2:
        transformation_matrix = np.expand_dims(transformation_matrix, 0)

    t = transformation_matrix[:,:3,3]
    Rs = transformation_matrix[:,:3,:3]

    R_rotmat = R.from_matrix(Rs)
    q = R_rotmat.as_quat()
    q = np.roll(q, 1, axis=1)


    qt = np.zeros((1,7))
    qt = np.repeat(qt, transformation_matrix.shape[0], 0)
    qt[:,:4] = q
    qt[:,4:] = t

    return qt

def quat_axis(q, axis=0):
    """ ?? """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)








