# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import torch
from transformers import AutoModel, AutoTokenizer
import time
from model.modeling_dream import DreamModel

import types
# Load model and tokenizer
device = "cuda:3"
# 从命令行读取use_cache
use_cache = True

if use_cache:
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()

    from model.generation_utils_block import DreamGenerationMixin
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)
else:
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device).eval()


# Initialize conversation history
messages = []

# Get user input
user_input = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

# Add user message to conversation history
messages.append({"role": "user", "content": user_input})

# Format input with chat template
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device=device)
attention_mask = inputs.attention_mask.to(device=device)

def generation_tokens_hook_func(step, x, logits):
    print(f"############ Step {step} ############")
    # print(tokenizer.decode(h[0].tolist()))
    print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "), end="\r")
    time.sleep(0.01)
    return x

# Generate response
start = time.time()
output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1024,
    output_history=True,
    return_dict_in_generate=True,
    steps=1024,
    temperature=0.,
    top_p=None,
    alg="entropy",
    alg_temp=0.1,
    top_k=None,
    block_length=32,
    method="EoT+dual_cache"
    # generation_tokens_hook_func=generation_tokens_hook_func
)
print(f"Time spent: {time.time() - start}")

# Process response
generation = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
generation = generation.split(tokenizer.eos_token)[0].strip()

# Print response
print("Model:", generation)