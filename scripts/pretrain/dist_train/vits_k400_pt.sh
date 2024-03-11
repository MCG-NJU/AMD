#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_s_k400_pt_800e'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='YOUR_PATH/data/k400/train.csv'  # The data list file path.
DATA_ROOT='YOUR_DATA_ROOT' # If the data list already contains absolute paths, then this can be empty.
TEACHER_PATH='VideoMAEv1_vit_L_k400_pt_1600e.pth' # Fill in the path to your teacher model.

N_NODES=${N_NODES:-4}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# batch_size can be adjusted according to the graphics card
python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_amd_pretrain.py \
        --dataset k400 \
        --data_root "${DATA_ROOT}" \
        --data_path ${DATA_PATH} \
        --mask_type t_consist  \
        --mask_ratio 0.90 \
        --mask_ratio_teacher 0.75 \
        --tubelet_size 2 \
        --model pretrain_mae_small_patch16_224 \
        --model_teacher vit_large_patch16_224 \
        --path_teacher ${TEACHER_PATH} \
        --decoder_depth 4 \
        --generator_depth 2 \
        --batch_size 16 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --save_ckpt_freq 50 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --lr 1.2e-3 \
        --student_layer_direct_align 6 12 \
        --student_layer_gen_align 6 12 \
        --teacher_layer_direct_align 12 24 \
        --teacher_layer_gen_align 12 24 \
        --num_workers 16 \
        --clip_grad 0.02 \
        --align_loss l2 \
        --output_dir ${OUTPUT_DIR}