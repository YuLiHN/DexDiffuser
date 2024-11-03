import os
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

# from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from utils.utils import save_ckpt, load_ckpt
import hydra
from omegaconf import DictConfig, OmegaConf
from models import create_ddpm, create_evaluator
from dataset import create_dataset_sampler, create_dataset_evaluator, collate_fn_general

from torch.utils.data.distributed import DistributedSampler


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ 
    Args:
        cfg: configuration dict
    """
    ## set rank
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(cfg.gpu)
    device = torch.device('cuda', cfg.gpu)
    torch.distributed.init_process_group(backend='nccl')

    ## set output logger and tensorboard
    ## Begin training progress
    if cfg.gpu == 0:
        mkdir_if_not_exists(cfg.tb_dir)
        mkdir_if_not_exists(cfg.vis_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)


        writer = SummaryWriter(log_dir=cfg.tb_dir)
        Ploter.setWriter(writer)

        ## Begin training progress
        logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
        logger.info('Begin training..')

    if cfg.task.dataset.name == 'MultidexSamplerAllegro' or cfg.task.dataset.name == 'DexGraspNetSamplerAllegro':
        datasets = {
            'train': create_dataset_sampler(cfg, 'train'),
        }
        model = create_ddpm(cfg)
        if cfg.gpu == 0:
            logger.info('training sampler!!!')
        
    elif cfg.task.dataset.name == 'EvaluatorDataset':
        datasets = {
            'train': create_dataset_evaluator(cfg, 'train'),
        }
        model = create_evaluator(cfg)
        if cfg.gpu == 0:
            logger.info('training evaluator!!!')

    
    ## convert to parallel
    model.to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True)
    model.train()

    if cfg.gpu == 0:
        for subset, dataset in datasets.items():
            logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    collate_fn = collate_fn_general
    
    train_sampler = DistributedSampler(datasets['train'])
    dataloaders = {
        'train': datasets['train'].get_dataloader(
            sampler=train_sampler,
            batch_size=cfg.task.train.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.train.num_workers,
            pin_memory=True,
        ),
    }
    
    
    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group) # use adam optimizer in default

    if cfg.gpu == 0:
        logger.info(f'{len(params)} parameters for optimization.')
        logger.info(f'total model size is {sum(nparams)}.')
    
    ## start training
    step = 0
    for epoch in range(0, cfg.task.train.num_epochs):

        for it, data in enumerate(dataloaders['train']):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            optimizer.zero_grad()
            data['epoch'] = epoch
            outputs = model(data)
            loss = outputs['loss'].mean()
            loss.backward()
            # outputs['loss'].backward()
            optimizer.step()
            
            ## plot loss
            if cfg.gpu == 0 and (step + 1) % cfg.task.train.log_step == 0:
                total_loss = loss #outputs['loss'].item()
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                for key in outputs:
                    # val = outputs[key].mean().item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'train/{key}': {'plot': True, 'value': total_loss, 'step': step},
                        'train/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })
            step += 1

        ## save ckpt in epoch
        if cfg.gpu == 0 and (epoch + 1) % cfg.save_model_interval == 0:
            save_path = os.path.join(
                cfg.ckpt_dir, 
                f'model_{epoch + 1}.pth' if cfg.save_model_seperately else 'model.pth'
            )

            save_ckpt(
                model=model, epoch=epoch+1, step=step, path=save_path,
                save_scene_model=True,
            )
        

    ## Training is over!
    if cfg.gpu == 0:
        writer.close() # close summarywriter and flush all data to disk
        logger.info('End training..')
        


if __name__ == '__main__':
    main()
