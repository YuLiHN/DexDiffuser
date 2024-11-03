from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer



class UNetModel(nn.Module):
    def __init__(self,cfg, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.model.d_x
        self.d_model = cfg.model.d_model
        self.nblocks = cfg.model.nblocks
        self.resblock_dropout = cfg.model.resblock_dropout
        self.transformer_num_heads = cfg.model.transformer_num_heads
        self.transformer_dim_head = cfg.model.transformer_dim_head
        self.transformer_dropout = cfg.model.transformer_dropout
        self.transformer_depth = cfg.model.transformer_depth
        self.transformer_mult_ff = cfg.model.transformer_mult_ff
        self.context_dim = cfg.model.context_dim
        self.use_position_embedding = cfg.model.use_position_embedding # for input sequence x
        self.scene_model_name = cfg.model.scene_model.name # 'obj_bps','random_condition', 'PointNet2'
        self.freeze_scene_model = cfg.model.freeze_scene_model

        scene_model_in_dim = 3
        scene_model_args = {'c': scene_model_in_dim, 'num_points': 2048}
        print("scene model name: ", self.scene_model_name)
        if self.scene_model_name == 'obj_bps':
            self.scene_model = None
        elif self.scene_model_name == 'random_condition':
            from bps_torch.bps import bps_torch
            import numpy as np
            basis_bps_set = np.load('./models/basis_point_set.npy')
            self.bps = bps_torch(n_bps_points=4096,
                    radius=1.,
                    n_dims=3,
                    custom_basis=basis_bps_set)
    
        elif self.scene_model_name == 'PointNet2':
            from models.model.pointnet2.pointnet2_semseg import pointnet2_enc_repro
            self.scene_model = pointnet2_enc_repro(**scene_model_args)
        else:
            raise NotImplementedError
        

        weight_path = None 
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
            
        

        time_embed_dim = self.d_model * cfg.model.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
            
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        elif self.scene_model_name == 'PointNet2':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)
        elif self.scene_model_name == 'obj_bps':
            b = data['obj_bps'].shape[0]
            scene_feat = data['obj_bps'].reshape(b, -1, self.context_dim)
        elif self.scene_model_name == 'random_condition':
            b = data['pos'].shape[0]
            scene_feat = self.bps.encode(torch.rand(b,2048,3),
                                   feature_type=['dists'])['dists'].reshape(b, -1, self.context_dim)
            # scene_feat = torch.randn(b,1,self.context_dim).to('cuda')
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat

def create_unet():
    return