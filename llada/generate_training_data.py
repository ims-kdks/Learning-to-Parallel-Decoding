from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from generate import get_num_transfer_tokens, add_gumbel_noise
from model.modeling_llada import LLaDAModelLM
import torch
import torch.nn.functional as F
import numpy as np
import accelerate
from tqdm import tqdm
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--num_sample", type=int, required=False, help="Number of samples to be selected from each task")
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--gen_length", type=int, required=True)
parser.add_argument("--block_length", type=int, required=True)
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()
print(args)

# create folder to store data for training
train_data_dir = "small_model_train/train_data"
os.makedirs(train_data_dir, exist_ok=True)

def get_subset(ds: Dataset):
    '''
    Select a subset of data from each of the tasks
    '''
    if args.num_sample is None:
        return ds
    df = ds.to_polars()
    df = df.group_by('task').map_groups(lambda x : x[:args.num_sample])
    return Dataset.from_polars(df)

def _tokenize(e):
    return {
        "question": tokenizer(e["inputs"])["input_ids"],
        "question_text": e["inputs"],
    }

@ torch.no_grad()
def generate_data(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, correct_answer=None, id=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    results = []
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        prev_data = None
        while (x[:, block_start:block_end] == mask_id).any():
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # generate training data by simulating Extremely Greedy Parallel process
            if correct_answer is not None:
                correct_block = correct_answer[:, block_start:block_end]
                block_confidence = x0_p[:, block_start:block_end]
                block_token = x0[:, block_start:block_end]
                is_correct = (block_token == correct_block)
                results.append({
                    'step': i,
                    'confidence': block_confidence.reshape(-1).to('cpu').tolist(),
                    'is_correct': is_correct.reshape(-1).to('cpu').tolist()
                })
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                # add this check because sometimes it will never predict correct answer and stuck in this loop
                if prev_data is not None and (block_token == prev_data).all():
                    x[:, block_start:block_end] = correct_block
                    continue
                transfer_index[:, block_start:block_end] = is_correct
                x[transfer_index] = x0[transfer_index]
                i += 1
                prev_data = block_token
                continue

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # unmask the highest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            i += 1
    
    # save data
    with open(f"{train_data_dir}/{id}.pkl", "wb") as file:
        pickle.dump(results, file)

    return x


accelerator = accelerate.Accelerator()

# prepar model
model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, dtype=torch.bfloat16, device_map=f'{accelerator.device}').eval()
# model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, dtype=torch.bfloat16, device_map='auto').eval()
model = accelerator.prepare(model)
device = torch.device(f'{accelerator.device}')

# prepar data
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
dataset = load_dataset("small_model_train/flan", split=args.split)
dataset = get_subset(dataset)
ds = dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
ds = ds.map(_tokenize).with_format('torch', device=device)

# generate data for training
for i, elem in enumerate(tqdm(ds)):
    prompt = elem['question'].unsqueeze(0).to(device)
    # get the correct answer first
    correct_answer = generate_data(model, prompt, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0, cfg_scale=0, remasking='low_confidence')

    _ = generate_data(model, prompt, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0, cfg_scale=0, remasking='low_confidence', correct_answer=correct_answer, id=f'{accelerator.process_index}-{i}')