#!/bin/bash
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUDA_VISIBLE_DEVICES=0,1,2,3


task=gsm8k
gen_length=256
block_length=32
steps=$((gen_length / block_length))
model="Dream-org/Dream-v0-Instruct-7B"
method=original
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${gen_length},diffusion_steps=${gen_length},add_bos_token=true,alg=entropy,show_speed=True,block_length=${block_length},method=${method} \
    --tasks ${task} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    > output/eval_results/$task/eval_${method}_${gen_length}.log 2>&1

sleep 3600

task=gsm8k
gen_length=256
block_length=32
steps=$((gen_length / block_length))
model="Dream-org/Dream-v0-Instruct-7B"
method=EoT
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${gen_length},diffusion_steps=${gen_length},add_bos_token=true,alg=entropy,show_speed=True,block_length=${block_length},method=${method} \
    --tasks ${task} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    > output/eval_results/$task/eval_${method}_${gen_length}.log 2>&1

# task=humaneval
# gen_length=256
# block_length=32
# steps=$((gen_length / block_length))
# model="Dream-org/Dream-v0-Base-7B"
# method=original

# # baseline
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${gen_length},diffusion_steps=${gen_length},add_bos_token=true,alg=entropy,show_speed=True,escape_until=true \
#     --tasks ${task} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     > output/eval_results/$task/eval_${method}_${gen_length}.log 2>&1
