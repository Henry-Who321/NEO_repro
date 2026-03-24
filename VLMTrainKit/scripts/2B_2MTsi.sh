#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# For detailed logic, refer to: neo/model/build.py build_model function
mllm="/vlm/hhr/NEO/VLMTrainKit/output/checkpoint-2000"  # Start from checkpoint-2000 weights
llm=""  # Path to the base LLM model for training NEO from scratch
tokenizer="/vlm/hhr/NEO/VLMTrainKit/output/checkpoint-2000"  # Path to the tokenizer

# Training hyperparameters
lr=5e-5

# Global batch size = batch_size * grad_accum_steps * num_gpus
batch_size=32  # Per-device batch size for controlling global batch size
grad_accum_steps=4  # Gradient accumulation steps for controlling global batch size

# Training entry point
entry_file=neo/train/train.py

# Dataset configuration (replace with public dataset names)
datasets="cambrian_737k"

# Output configuration
run_name=neo-baseline-PT_2B_2MTsi
output_dir=./output_2MTsi   # MUST differ from ./output to avoid picking up ZeRO-2 checkpoints as resume targets
timestamp=$(date +"%Y%m%d_%H%M%S")
tb_log_dir=${TB_LOG_DIR:-${output_dir}/runs/${run_name}_${timestamp}}
log_dir=${output_dir}/logs
log_file=${log_dir}/${run_name}_${timestamp}.log
tbdev_upload=${TBDEV_UPLOAD:-0}
tbdev_name=${TBDEV_NAME:-${run_name}_${timestamp}}
tbdev_description=${TBDEV_DESCRIPTION:-"NEO 2B 2MT training logs"}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${mllm} \
    --tokenizer_name_or_path ${tokenizer} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --bf16 True \
    --output_dir ${output_dir} \
    --extra_num_layers 12 \
    --num_hidden_layers 40 \
    --max_steps 50000 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 16777216 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 25 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --min_lr_ratio 0.2 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --max_seq_length 32768 \
    --model_max_length 32768 \
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
# Usage: TBDEV_UPLOAD=1 bash scripts/2B_2MTsi.sh
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