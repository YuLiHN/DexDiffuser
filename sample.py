import os
# import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from utils.utils import load_ckpt
from utils.io import mkdir_if_not_exists
import hydra
from models import create_ddpm, create_evaluator, create_visualizer

@hydra.main(version_base=None, config_path="./configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    vis_dir = cfg.eval_dir + '/' + cfg.exp_dir +  '/' + f'{cfg.dataset_name}'
    mkdir_if_not_exists(vis_dir)

    logger.add(vis_dir + '/sample.log') # set logger file
    # logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg)) # record configuration

    if cfg.gpu is not None:
        device=f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    ## create model and load ckpt
    model = create_ddpm(cfg)
    model.to(device=device)
    model.eval()
    
    L=[10,4,-1]
    evaluator = create_evaluator(cfg,pos_enc_multires=L)
    # evaluator = create_evaluator(cfg)
    evaluator.to(device=device)
    evaluator.eval()
    
    if cfg.model.scene_model.name == 'obj_bps':
        sampler_pth = cfg.sampler_bps_ckpt_pth
    elif cfg.model.scene_model.name == 'PointNet2':
        sampler_pth = cfg.sampler_pn2_ckpt_pth
    else:
        raise NotImplementedError
    ## if your models are seperately saved in each epoch, you need to change the model path manually
    load_ckpt(model, path=sampler_pth)
    load_ckpt(evaluator, path=cfg.evaluator_ckpt_pth)
    
    ## create visualizer and visualize
    visualizer = create_visualizer(cfg, scale=True)
    visualizer.sample_grasps(model, 
                         cfg.dataset_name,
                         cfg.data_root, 
                         vis_dir, 
                         cam_views=cfg.cam_views, 
                         evaluator=evaluator, 
                         guid_scale=cfg.guid_scale, 
                         num_sample=cfg.num_sample,
                         vis_type=None)
    logger.info('done!') # set logger file

if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()