export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./cola_delora"

lora_r_values="128"
for lora_r in $lora_r_values; do

    lora_alpha=$((2 * $lora_r))

    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name cola \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 4e-4 \
    --num_train_epochs 80 \
    --output_dir $output_dir/$lora_r/model \
    --overwrite_output_dir \
    --logging_steps 100 \
    --logging_dir $output_dir/log \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --warmup_ratio 0.06 \
    --apply_lora \
    --apply_delora \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --seed 0 \
    --weight_decay 0.1
done
