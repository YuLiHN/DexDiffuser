from .sampler_dataset import DexGraspNetSamplerAllegro
from .evaluator_dataset import DexGraspNetEvaluatorDataset
from .misc import collate_fn_general

def create_dataset_sampler(cfg, mode):

    dataset = DexGraspNetSamplerAllegro(mode, cfg)
    return dataset

def create_dataset_evaluator(cfg, mode):

    dataset = DexGraspNetEvaluatorDataset(cfg, mode)
    return dataset
