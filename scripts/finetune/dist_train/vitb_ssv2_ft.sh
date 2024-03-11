#!/usr/bin/env bash
set -x

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_ssv2_pt_800e_ssv2_ft'
DATA_PATH='YOUR_PATH/data/ssv2'
DATA_ROOT='YOUR_DATA_ROOT' # If the data list already contains absolute paths, then this can be empty.
MODEL_PATH='YOUR_PATH/model_zoo/vit_b_ssv2_pt_800e.pth'

N_NODES=${N_NODES:-2}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training argsl

python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set SSV2 \
        --data_root "${DATA_ROOT}" \
        --nb_classes 174 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --opt adamw \
        --lr 5e-4 \
        --num_workers 16 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 30 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --enable_deepspeed \
        ${PY_ARGS}