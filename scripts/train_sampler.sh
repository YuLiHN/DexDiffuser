EXP_NAME=dexdiffuser

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                diffuser=ddpm \
                diffuser.loss_type=l1 \
                diffuser.steps=100 \
                model=unet_grasp_bps \
                task=grasp_gen_ur_dexgn_slurm \
