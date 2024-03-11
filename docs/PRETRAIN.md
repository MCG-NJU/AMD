# Pre-training AMD

Our codebase supports **multi-node slurm training** and **multi-node distributed training**. We provide the **off-the-shelf** slurm training scripts in the [pre-train scripts folder](/scripts/pretrain). Below we give an example of the pre-training script.

## Slurm Train

To pre-train AMD ViT-B on Kinetics-400 with 32 A100-80G (4 nodes x 8 GPUs), you can use the following script file **script/pretrain/slurm_train/vitb_k400_pt.sh**.

```bash
#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_k400_pt_800e_k400_ft'
DATA_PATH='YOUR_PATH/data/k400'
DATA_ROOT='YOUR_DATA_ROOT' # If the data list already contains absolute paths, then this can be empty.
MODEL_PATH='YOUR_PATH/model_zoo/vit_b_k400_pt_800e.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set K400 \
        --data_root "${DATA_ROOT}" \
        --nb_classes 400 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 30 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 7e-4 \
        --num_workers 16 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 90 \
        --dist_eval \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --enable_deepspeed \
        ${PY_ARGS}
  ```

Start training by running
```bash
bash scripts/pretrain/slurm_train/vitb_k400_pt.sh k400_pretrain
```
, where `k400_pretrain` is the job name.

## Dist Train

The above slurm training script can be modified to distributed training script as follows:

```bash
#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_k400_pt_800e'  # Your output folder for deepspeed config file, logs and checkpoints
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
        --model pretrain_mae_base_patch16_224 \
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
```
Start training by run
```bash
NODE_RANK=0  # 0 for the first node, 1 for the second node, and so on.
# MASTER_ADDR should be set as the ip of current node

bash scripts/pretrain/dist_train/vitb_k400_pt.sh $NODE_RANK $MASTER_ADDR
# bash scripts/pretrain/dist_train/vitb_k400_pt.sh 0 127.0.0.1
```
at each node. 