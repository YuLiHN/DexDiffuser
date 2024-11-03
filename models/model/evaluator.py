from re import T
import time

from flask.cli import F
import torch
# from FFHNet.utils import utils
from torch import nn
from torch.optim import lr_scheduler
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
# from FFHNet.models import losses
from models.model.utils import get_embedder 

class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout




class DexEvaluator(nn.Module):
    def __init__(self,
                 cfg=None,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=6 + 3,
                 dtype=torch.float32,
                 device = 'cuda',
                 pos_enc_multires = None,
                 **kwargs):
        super(DexEvaluator, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pos_enc_multires = pos_enc_multires
        self.use_bn = False
        self.use_drop_out = True

        if pos_enc_multires is None:
            in_dim = in_bps + in_pose + 16
        else:
            self.embed_fn = [] 
            out_dim_embd = 0
            x_dims = [3,6,16]
            for i,pos_enc in enumerate(pos_enc_multires):
                embed_fn, out_dim = get_embedder(pos_enc, in_dim=x_dims[i])
                out_dim_embd += out_dim
                self.embed_fn.append(embed_fn)
            in_dim = in_bps + out_dim_embd
            
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(in_dim)
        self.rb1 = ResBlock(in_dim, n_neurons)
        self.rb2 = ResBlock(in_dim + n_neurons, n_neurons)
        self.rb3 = ResBlock(in_dim + n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 1)
        self.dout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.dtype = dtype

        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

    def compute_loss(self, pred_success_p, gt_label):
        """
            Computes the binary cross entropy loss between predicted success-label and true success
        """
        bce_loss_val = 10. * self.BCE_loss(pred_success_p, gt_label)
        return bce_loss_val
        
    
    def adapt_rot6d_to_9d(self,x):
        rot6d = x[:,3:9]
        rot9d = robust_compute_rotation_matrix_from_ortho6d(rot6d).view(-1,9)
        eva_input = torch.cat([rot9d, x[:,:3], x[:,9:]], dim=1)
        return eva_input

    def forward(self, data):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, obj_bps,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """

        if 'label' in data.keys():
            gt_label = data["label"].to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        
        if self.pos_enc_multires is None:
            X = torch.cat([data['obj_bps'], data['x_t']], dim=1).to(dtype=self.dtype, device=self.device).contiguous()
        else:
            embedded_trans = self.embed_fn[0](data['x_t'][:,:3]).to(dtype=self.dtype, device=self.device)
            embedded_rot = self.embed_fn[1](data['x_t'][:,3:9]).to(dtype=self.dtype, device=self.device)
            embedded_joint = self.embed_fn[2](data['x_t'][:,9:]).to(dtype=self.dtype, device=self.device)
            X = torch.cat([data['obj_bps'], embedded_trans, embedded_rot, embedded_joint], dim=1).to(dtype=self.dtype, device=self.device).contiguous()
        

        #X0 = self.bn1(X)
        if self.use_bn:
            X0 = self.bn1(X)
        else:
            X0=X
        X = self.rb1(X)
        if self.use_drop_out:
            X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        if self.use_drop_out:
            X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))
        if self.use_drop_out:
            X = self.dout(X)
        X = self.out_success(X)

        p_success = self.sigmoid(X)

        if 'label' in data.keys():
            loss = self.compute_loss(p_success, gt_label)
        else:
            loss = None

        out_dict = {
            'p_success': p_success,
            'loss': loss
        }

        return out_dict

