EXP_NAME=$1


CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 --master_port=25678 --use_env train_ddm.py \
                                        exp_name=${EXP_NAME} \
                                        save_model_interval=50 \
                                        diffuser=ddpm \
                                        diffuser.loss_type=l1 \
                                        diffuser.steps=100 \
                                        model=unet_grasp_bps \
                                        task=grasp_gen_ur_dexgn_slurm 