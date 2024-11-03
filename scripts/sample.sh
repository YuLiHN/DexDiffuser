MODEL=$1 #['bps', 'pn2']

python sample.py hydra/job_logging=none hydra/hydra_logging=none \
        diffuser=ddpm \
        diffuser.loss_type=l1 \
        diffuser.steps=100 \
        model=unet_grasp_${MODEL} \
        task=grasp_gen_ur_dexgn_slurm \
        task.dataset.normalize_x=true \
        task.dataset.normalize_x_trans=false \
        exp_name=bps \
        dataset_name=multidex \
