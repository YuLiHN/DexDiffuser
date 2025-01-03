import json
import os
import pytorch_kinematics as pk
import torch.nn
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.rot6d import *
import trimesh.sample
import pickle

_joint_angle_lower = torch.tensor([-0.47, -0.196, -0.174, -0.227, -0.47, -0.196, -0.174,
                                                -0.227, -0.47, -0.196, -0.174, -0.227, 0.263, -0.105,
                                                -0.189, -0.162])
_joint_angle_upper = torch.tensor([0.47, 1.61, 1.709, 1.618, 0.47, 1.61, 1.709, 1.618,
                                                0.47, 1.61, 1.709, 1.618, 1.396, 1.163, 1.644, 1.719])

_global_trans_lower = torch.tensor([-0.17810515, -0.2110989,  -0.19037187])
_global_trans_upper = torch.tensor([0.17987582, 0.18825185, 0.18027663])

_NORMALIZE_LOWER = -1.
_NORMALIZE_UPPER = 1.

class HandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 hand_scale=2.
                 ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)
        # prepare contact point basis and surface point samples
        # self.no_contact_dict = json.load(open(os.path.join('data', 'urdf', 'intersection_%s.json'%robot_name)))

        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        # prepare contact point basis and surface point samples
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []
        self.link_pcd_dict = {}

        # self.point_on_hand = pickle.load(open('./link_pcd_dict.pickle', 'rb'))
        verts_bias = 0

        if robot_name == 'shadowhand':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4).repeat(self.batch_size, 1, 1)
        elif robot_name == 'allegro' or 'allegro_right':
            self.palm_toward = torch.tensor([0., -1., 0., 0.], device=self.device).reshape(1, 1, 4).repeat(self.batch_size, 1, 1)
            
        else:
            raise NotImplementedError

        for i_link, link in enumerate(visual.links):
            # print(f"Processing link #{i_link}: {link.name}")
            # load mesh
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                # print(link.visuals[0])
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett' or robot_name == 'allegro_right':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'allegro':
                    filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            # Surface point
            # mesh.sample(int(mesh.area * 100000)) * scale
            # todo: marked original count is 128
            if self.robot_name == 'shadowhand':
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            if self.robot_name == 'barrett':
                if link.name in ['bh_base_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'ezgripper':
                if link.name in ['left_ezgripper_palm_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[1., 0., 0.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'robotiq_3finger':
                if link.name in ['gripper_palm']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)

            pts *= scale
            # pts = mesh.sample(128) * scale
            # print(link.name, len(pts))
            # new
            if robot_name == 'shadowhand':
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            # pts_normal = np.matmul(rotation, pts_normal.T).T
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == 'shadowhand':
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute':
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.revolute_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_lower = torch.Tensor(
            self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(device)
        self.revolute_joints_q_upper = torch.Tensor(
            self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(device)

        self.current_status = None

        self.scale = hand_scale

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation.float(), surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_palm_points(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        if self.robot_name == 'shadowhand':
            link_name = 'palm'
        elif self.robot_name == 'allegro' or self.robot_name == 'allegro_right':
            link_name = 'base_link'
        
            # for link_name in parts:
            # get transformation
        trans_matrix = self.current_status[link_name].get_matrix()
        surface_points.append(
            torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale

    def get_palm_toward_point(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        # link_name = 'palm'
        if self.robot_name == 'shadowhand':
            link_name = 'palm'
        elif self.robot_name == 'allegro' or self.robot_name == 'allegro_right':
            link_name = 'base_link'
        trans_matrix = self.current_status[link_name].get_matrix()
        palm_toward_point = torch.matmul(trans_matrix, self.palm_toward.transpose(1, 2)).transpose(1, 2)[..., :3]
        palm_toward_point = torch.matmul(self.global_rotation, palm_toward_point.transpose(1, 2)).transpose(1, 2)

        return palm_toward_point.squeeze(1)

    def get_palm_center_and_toward(self, q=None):
        if q is not None:
            self.update_kinematics(q)

        palm_surface_points = self.get_palm_points()
        palm_toward_point = self.get_palm_toward_point()

        palm_center_point = torch.mean(palm_surface_points, dim=1, keepdim=False)
        return palm_center_point, palm_toward_point

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            surface_normals.append(
                torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[...,
                :3])
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1,
                                                                                                      2) + self.global_translation.unsqueeze(
            1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)

        return surface_points * self.scale, surface_normals

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data
    
    
    


    


 
def get_handmodel(batch_size, device, hand_scale=1., urdf_path = './data/urdf', robot='allegro_right'):

    # assets_meta_path = os.path.join(urdf_path,'urdf_assets_meta.json')
    # urdf_assets_meta = json.load(open(assets_meta_path))
    # urdf_path = urdf_assets_meta['urdf_path'][robot]
    # meshes_path = urdf_assets_meta['meshes_path'][robot]
    if robot == 'allegro_right':
        urdf_file = 'allegro_hand_description/allegro_hand_description_right.urdf'
    elif robot == 'allegro_left':
        urdf_file = 'allegro_hand_description/allegro_hand_description_left.urdf'
    else:
        raise NotImplementedError
    meshes_path = os.path.join(urdf_path, 'allegro_hand_description/meshes')
    hand_urdf_path = os.path.join(urdf_path, urdf_file)
    hand_model = HandModel(robot, hand_urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale)
    return hand_model


def compute_collision(obj_pcd_nor: torch.Tensor, hand_pcd: torch.Tensor):
    """
    :param obj_pcd_nor: N_obj x 6
    :param hand_surface_points: B x N_hand x 3
    :return:
    """
    b = hand_pcd.shape[0]
    n_obj = obj_pcd_nor.shape[0]
    n_hand = hand_pcd.shape[1]

    obj_pcd = obj_pcd_nor[:, :3]
    obj_nor = obj_pcd_nor[:, 3:6]

    # batch the obj pcd
    batch_obj_pcd = obj_pcd.unsqueeze(0).repeat(b, 1, 1).view(b, 1, n_obj, 3)
    batch_obj_pcd = batch_obj_pcd.repeat(1, n_hand, 1, 1)
    # batch the hand pcd
    batch_hand_pcd = hand_pcd.view(b, n_hand, 1, 3).repeat(1, 1, n_obj, 1)
    # compute the pair wise dist
    hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
    hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
    # gather the obj points and normals w.r.t. hand points
    hand_obj_points = torch.stack([obj_pcd[x, :] for x in hand_obj_indices], dim=0)
    hand_obj_normals = torch.stack([obj_nor[x, :] for x in hand_obj_indices], dim=0)
    # compute the signs
    hand_obj_signs = ((hand_obj_points - hand_pcd) * hand_obj_normals).sum(dim=2)
    # hand_obj_signs = (hand_obj_signs > 0.).float()
    hand_obj_signs = torch.sign(hand_obj_signs) # negative for no collision, positive for collision
    # signs dot dist to compute collision value
    collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
    # collision_value = (hand_obj_signs * hand_obj_dist).mean(dim=1)
    return collision_value

def trans_normalize(global_trans: torch.Tensor):
    
    device = global_trans.device
    
    global_trans_norm = torch.div((global_trans - _global_trans_lower.to(device)), 
                                  (_global_trans_upper.to(device) - _global_trans_lower.to(device)))
    global_trans_norm = global_trans_norm * (_NORMALIZE_UPPER - _NORMALIZE_LOWER) - (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    return global_trans_norm

def trans_denormalize(global_trans: torch.Tensor):
    
    device = global_trans.device
    
    global_trans_denorm = global_trans + (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    global_trans_denorm /= (_NORMALIZE_UPPER - _NORMALIZE_LOWER)
    global_trans_denorm = global_trans_denorm * (_global_trans_upper.to(device) - _global_trans_lower.to(device)) + _global_trans_lower.to(device)
    return global_trans_denorm

def angle_normalize(joint_angle: torch.Tensor):
    
    device = joint_angle.device
    
    joint_angle_norm = torch.div((joint_angle - _joint_angle_lower.to(device)), (_joint_angle_upper.to(device) - _joint_angle_lower.to(device)))
    joint_angle_norm = joint_angle_norm * (_NORMALIZE_UPPER - _NORMALIZE_LOWER) - (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    return joint_angle_norm

def angle_denormalize(joint_angle: torch.Tensor):
    
    device = joint_angle.device
    
    joint_angle_denorm = joint_angle + (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    joint_angle_denorm /= (_NORMALIZE_UPPER - _NORMALIZE_LOWER)
    joint_angle_denorm = joint_angle_denorm * (_joint_angle_upper.to(device) - _joint_angle_lower.to(device)) + _joint_angle_lower.to(device)
    return joint_angle_denorm


if __name__ == '__main__':
    from plotly_utils import plot_point_cloud
    seed = 0
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_model = get_handmodel(1, 'cuda')
    print(len(hand_model.robot.get_joint_parameter_names()))

    joint_lower = np.array(hand_model.revolute_joints_q_lower.cpu().reshape(-1))
    joint_upper = np.array(hand_model.revolute_joints_q_upper.cpu().reshape(-1))
    print(joint_lower, joint_upper)
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    q = torch.from_numpy(np.concatenate([np.array([0.06, 0.10, -0.02, 0.51, 0.01, 0.35, -0.2, 0.66, 0.34]), joint_mid])).unsqueeze(0).to(
        device).to(torch.float32)
    data = hand_model.get_plotly_data(q=q, opacity=0.5)
    palm_center_point, palm_toward_point = hand_model.get_palm_center_and_toward()
    # data.append(plot_point_cloud(palm_toward_point.cpu() + palm_center_point.cpu(), color='black'))
    data.append(plot_point_cloud(palm_center_point.cpu(), color='red'))
    fig = go.Figure(data=data)
    fig.show()
