#!/bin/bash

# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
# For detailed logic, refer to: neo/model/build.py build_model function
mllm=""  # Path to pre-trained NEO model for SFT (Supervised Fine-Tuning) on top of an existing checkpoint
llm="/vlm/pretrain_models/Qwen3/Qwen3-1.7B-Base"  # Path to the base LLM model for training NEO from scratch
tokenizer="/vlm/pretrain_models/Qwen3/Qwen3-1.7B-Base"  # Path to tokenizer directory or HF repo id

# Training hyperparameters
lr=8e-4
# Global batch size = batch_size * grad_accum_steps * num_gpus
batch_size=16  # Per-device batch size for controlling global batch size
grad_accum_steps=4  # Gradient accumulation steps for controlling global batch size

# Training entry point
entry_file=neo/train/train.py

# Dataset configuration (replace with public dataset names)
datasets="cambrian_737k"

# Output configuration
run_name=neo-baseline-PT_2B
output_dir=./output
timestamp=$(date +"%Y%m%d_%H%M%S")
tb_log_dir=${TB_LOG_DIR:-${output_dir}/runs/${run_name}_${timestamp}}
log_dir=${output_dir}/logs
log_file=${log_dir}/${run_name}_${timestamp}.log
tbdev_upload=${TBDEV_UPLOAD:-0}
tbdev_name=${TBDEV_NAME:-${run_name}_${timestamp}}
tbdev_description=${TBDEV_DESCRIPTION:-"NEO 2B 1PT training logs"}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --llm_model_name_or_path ${llm} \
    --tokenizer_name_or_path ${tokenizer} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --bf16 True \
    --output_dir ${output_dir} \
    --extra_num_layers 12 \
    --num_hidden_layers 32 \
    --train_buffer True \
    --max_steps 5000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 4194304 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 25 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --min_lr_ratio 0.1 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --max_seq_length 16384 \
    --model_max_length 8192 \
    --patch_size 16 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --logging_dir ${tb_log_dir} \
    --report_to tensorboard"

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Save terminal logs to file while keeping real-time console output.
mkdir -p "${log_dir}"
mkdir -p "${tb_log_dir}"
exec > >(tee -a "${log_file}") 2>&1
echo "[INFO] Training log file: ${log_file}"
echo "[INFO] TensorBoard log dir: ${tb_log_dir}"

# Launch training
train_exit_code=0
torchrun --nproc_per_node=4 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} || train_exit_code=$?

# Optional: one-click upload to TensorBoard.dev (Google account).
# Usage: TBDEV_UPLOAD=1 bash scripts/2B_1PT.sh
if [ "${tbdev_upload}" = "1" ]; then
    echo "[INFO] TensorBoard.dev upload enabled."
    if ! command -v tensorboard >/dev/null 2>&1; then
        echo "[ERROR] tensorboard command not found. Install with: pip install tensorboard"
    elif [ ! -d "${tb_log_dir}" ]; then
        echo "[ERROR] TensorBoard log dir not found: ${tb_log_dir}"
    else
        echo "[INFO] Uploading logs to TensorBoard.dev (first run requires browser login)..."
        tensorboard dev upload \
            --logdir "${tb_log_dir}" \
            --name "${tbdev_name}" \
            --description "${tbdev_description}" \
            --one_shot
    fi
fi

exit ${train_exit_code}