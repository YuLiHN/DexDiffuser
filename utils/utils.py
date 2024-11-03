import torch
import os
from loguru import logger

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'
    if path.split('.')[-1] == 'pth':
        saved_state_dict = torch.load(path)['model']
    else:
        saved_state_dict = torch.load(path)['ffhevaluator_state_dict']
    model_state_dict = model.state_dict()
    total = 0
    
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            nparams.append(p.nelement())
            # logger.info(f'add {n} {p.shape} for optimization')
    # logger.info(f'{len(params)} parameters for optimization.')
    logger.info(f'total model size is {sum(nparams)}.')

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            # logger.info(f'Load parameter {key} for current model.')
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
    
    model.load_state_dict(model_state_dict)

def save_ckpt(model: torch.nn.Module, epoch: int, step: int, path: str, save_scene_model: bool) -> None:
    """ Save current model and corresponding data

    Args:
        model: best model
        epoch: best epoch
        step: current step
        path: save path
        save_scene_model: if save scene_model
    """
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        ## if use frozen pretrained scene model, we can avoid saving scene model to save space
        if 'scene_model' in key and not save_scene_model:
            continue

        saved_state_dict[key] = model_state_dict[key]
    
    logger.info('Saving model!!!' + ('[ALL]' if save_scene_model else '[Except SceneModel]'))
    torch.save({
        'model': saved_state_dict,
        'epoch': epoch, 'step': step,
    }, path)