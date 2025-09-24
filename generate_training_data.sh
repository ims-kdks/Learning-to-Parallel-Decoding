#!/bin/bash
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

accelerate launch generate_training_data.py \
    --steps 256 \
    --gen_length 256 \
    --block_length 32 \
    --split train \
    --num_sample 40