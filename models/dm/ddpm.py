from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from utils.handmodel import angle_denormalize, trans_denormalize
from models.dm.schedule import make_schedule_ddpm
import numpy as np
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d, compute_pitch

# @DIFFUSER.register()
class DDPM(nn.Module):
    def __init__(self,cfg:DictConfig, eps_model: nn.Module, has_obser: bool, *args, **kwargs) -> None:
        super(DDPM, self).__init__()
        
        self.eps_model = eps_model
        self.timesteps = cfg.diffuser.steps
        self.schedule_cfg = cfg.diffuser.schedule_cfg
        self.rand_t_type = cfg.diffuser.rand_t_type

        self.has_observation = has_obser # used in some task giving observation

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        
        if cfg.diffuser.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.diffuser.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
                
        # self.handmodel = get_handmodel(1, 'cuda', urdf_path=cfg.task.dataset.urdf_root)
        # self.normalize_x = cfg.task.dataset.normalize_x
        # self.normalize_x_trans = cfg.task.dataset.normalize_x_trans

    @property
    def device(self):
        return self.betas.device
    
    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Apply observation to x_t, if self.has_observation if False, this method will return the input

        Args:
            x_t: noisy x in step t
            data: original data provided by dataloader
        """
        ## has start observation, used in path planning and start-conditioned motion generation
        if self.has_observation and 'start' in data:
            start = data['start'] # <B, T, D>
            T = start.shape[1]
            x_t[:, 0:T, :] = start[:, 0:T, :].clone()
        
            if 'obser' in data:
                obser = data['obser']
                O = obser.shape[1]
                x_t[:, T:T+O, :] = obser.clone()
        
        return x_t
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        B = data['x'].shape[0]

        ## randomly sample timesteps
        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data)

        ## predict noise
        condtion = self.eps_model.condition(data)
        output = self.eps_model(x_t, ts, condtion)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        output = self.apply_observation(output, data)

        ## calculate loss
        loss = self.criterion(output, noise)

        return {'loss': loss}
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        pred_noise = self.eps_model(x_t, t, cond)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance
    
    def cond_fn(self, x_t, t, obj_bps, evaluator, guid_scale):
        
        # x_c = x_t.clone()
        # if self.normalize_x:
        #     # Compute the angle-denormalized part and store it in a new variable
        #     angle_denormalized_part = angle_denormalize(joint_angle=x_t[:, 9:])

        #     # Concatenate the unmodified and angle-denormalized parts
        #     x_t = torch.cat([x_t[:, :9], angle_denormalized_part], dim=1)

        # if self.normalize_x_trans:
        #     # Compute the trans-denormalized part and store it in a new variable
        #     trans_denormalized_part = trans_denormalize(global_trans=x_t[:, :3])

        #     # Concatenate the trans-denormalized part with the rest of x_t_clone
        #     x_t = torch.cat([trans_denormalized_part, x_t[:, 3:]], dim=1)
        
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            p_success = evaluator({'x_t':x_in,
                                   'obj_bps':obj_bps})['p_success']
            p_success_clip = torch.clamp(p_success, 1e-5, 1-1e-5)
            c_energy = torch.mean(torch.log(p_success_clip)) * guid_scale * np.log(self.timesteps - t + 1)
            
        return torch.autograd.grad(c_energy, x_in)[0]

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict, guid_param:Dict=None) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        if guid_param is not None:
            model_mean = model_mean + self.cond_fn(x_t, t, data['obj_bps'], guid_param['evaluator'], guid_param['guid_scale']) * model_variance
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x

    @torch.no_grad()
    def p_sample_loop(self, data: Dict, guid_param: Dict = None) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """

        # TODO: add classifier, remember the forward kinematic and denormalization, also check the gradient pass
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        x_t = self.apply_observation(x_t, data)
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data, guid_param)
            ## apply observation to x_t
            x_t = self.apply_observation(x_t, data)
            
            all_x_t.append(x_t)
        return torch.stack(all_x_t, dim=1)
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int=1, guid_param: Dict = None) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data, guid_param))
        
        ksamples = torch.stack(ksamples, dim=1)
        
        ## for sequence, normalize and convert repr
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                ## the start observation frames are replace during sampling
                _, O, _ = data['start'].shape
            ksamples[..., O:, :] = data['normalizer'].unnormalize(ksamples[..., O:, :])
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                O = 1
                if self.has_observation and 'start' in data:
                    _, O, _ = data['start'].shape
                ksamples[..., O-1:, :] = torch.cumsum(ksamples[..., O-1:, :], dim=-2)
            else:
                raise Exception('Unsupported repr type.')
        
        return ksamples