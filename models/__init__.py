from .model.unet import UNetModel
from .dm.ddpm import DDPM
from .model.visualizer import GraspGenURVisualizer
from .model.evaluator import DexEvaluator

def create_unet(cfg):
    return UNetModel(cfg)

def create_ddpm(cfg):
    return DDPM(cfg, create_unet(cfg),False)

def create_evaluator(cfg=None,pos_enc_multires=None):
    return DexEvaluator(cfg,pos_enc_multires=pos_enc_multires)

def create_visualizer(cfg,scale=False):

    return GraspGenURVisualizer(cfg)