#!/bin/bash
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
gen_length=256
block_length=32
method=original
accelerate launch eval_llada.py --tasks $task --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method \
> output/eval_results/$task/eval_${method}_${gen_length}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}.log"

task=gsm8k
gen_length=256
block_length=32
method=EoT
accelerate launch eval_llada.py --tasks $task --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method \
> output/eval_results/$task/eval_${method}_${gen_length}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}.log"

task=gsm8k
gen_length=256
block_length=32
method=L2P
accept_thres=0.96
accelerate launch eval_llada.py --tasks ${task} --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method,accept_thres=$accept_thres \
> output/eval_results/$task/eval_${method}_${gen_length}_${accept_thres//./}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}_${accept_thres//./}.log"

task=gsm8k
gen_length=256
block_length=32
method=L2P+EoT
accept_thres=0.96
accelerate launch eval_llada.py --tasks ${task} --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method,accept_thres=$accept_thres \
> output/eval_results/$task/eval_${method}_${gen_length}_${accept_thres//./}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}_${accept_thres//./}.log"

task=gsm8k
gen_length=256
block_length=32
method=L2P+EoT+dual_cache
accept_thres=0.96
accelerate launch eval_llada.py --tasks ${task} --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method,accept_thres=$accept_thres \
> output/eval_results/$task/eval_${method}_${gen_length}_${accept_thres//./}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}_${accept_thres//./}.log"

task=gsm8k
gen_length=256
block_length=32
method=L2P+EoT+prefix_cache
accept_thres=0.96
accelerate launch eval_llada.py --tasks ${task} --model llada_dist --confirm_run_unsafe_code \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method,accept_thres=$accept_thres \
> output/eval_results/$task/eval_${method}_${gen_length}_${accept_thres//./}.log 2>&1
echo "Finished: $task/eval_${method}_${gen_length}_${accept_thres//./}.log"