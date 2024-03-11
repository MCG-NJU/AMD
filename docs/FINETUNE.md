# Fine-tuning AMD

Our codebase supports **multi-node slurm training** and **multi-node distributed training**. We provide the **off-the-shelf** slurm training scripts in the [fine-tune scripts folder](/scripts/finetune). Below we give an example of the fine-tuning script.

## Slurm Train

To fine-tune AMD ViT-B on Kinetics-400 with 32 A100-80G (4 nodes x 8 GPUs), you can use the following script file **scripts/finetune/slurm_train/vitb_k400_ft.sh**.

```bash
#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000)) # Randomly set master_port to avoid port conflicts
export OMP_NUM_THREADS=1 # Control the number of threads

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_k400_pt_800e_k400_ft' # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='YOUR_PATH/data/k400' # The data list folder. the folder has three files: train.csv, val.csv, test.csv
DATA_ROOT='YOUR_DATA_ROOT' # If the data list already contains absolute paths, then this can be empty.
MODEL_PATH='YOUR_PATH/model_zoo/vit_b_k400_pt_800e.pth' # Model for initializing parameters

JOB_NAME=$1  # the job name of the slurm task
PARTITION=${PARTITION:-"video"}  # Name of the partition
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-32}  # Number of GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
CPUS_PER_TASK=${CPUS_PER_TASK:-16}  # Number of CPU cores allocated, number of tasks equal to the number of GPUs used
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

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
bash scripts/finetune/slurm_train/vitb_k400_ft.sh k400_finetune
```
, where `k400_finetune` is the job name.

If you just want to **test the performance of the model**, change `MODEL_PATH` to the model to be tested, `OUTPUT_DIR` to the path of the folder where the test results are saved, and run the following command:
```bash
bash scripts/finetune/slurm_train/vitb_k400_ft.sh --eval
```

## Dist Train

The above slurm training script can be modified to distributed training script as follows:

```bash
#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export OMP_NUM_THREADS=1

OUTPUT_DIR='YOUR_PATH/work_dir/vit_b_k400_pt_800e_k400_ft' # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='YOUR_PATH/data/k400' # The data list folder. the folder has three files: train.csv, val.csv, test.csv
DATA_ROOT='YOUR_DATA_ROOT' # If the data list already contains absolute paths, then this can be empty.
MODEL_PATH='YOUR_PATH/model_zoo/vit_b_k400_pt_800e.pth' # Model for initializing parameters

N_NODES=${N_NODES:-4}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# batch_size can be adjusted according to the graphics card
python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_class_finetuning.py \
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
Start training by run
```bash
NODE_RANK=0  # 0 for the first node, 1 for the second node, and so on.
# MASTER_ADDR should be set as the ip of current node

bash scripts/finetune/dist_train/vitb_k400_ft.sh $NODE_RANK $MASTER_ADDR
```
at each node. 