import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import accelerate
from tqdm import tqdm
import argparse
import pickle
from model.modeling_dream import DreamModel
from model.generation_utils_block import sample_tokens


parser = argparse.ArgumentParser()
parser.add_argument("--num_sample", type=int, required=False, help="Number of samples to be selected from each task")
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--gen_length", type=int, required=True)
parser.add_argument("--block_length", type=int, required=True)
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()
print(args)

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
    inputs = tokenizer(e["inputs"], return_tensors="pt", padding=True, padding_side="left")
    return {
        "question": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "question_text": e["inputs"],
    }

accelerator = accelerate.Accelerator()

# prepar model
model_path = "Dream-org/Dream-v0-Instruct-7B"
model = DreamModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map=f'{accelerator.device}').eval()
model = accelerator.prepare(model)
# model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
# model._sample = types.MethodType(DreamGenerationMixin._sample, model)
device = torch.device(f'{accelerator.device}')

# prepar data
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
dataset = load_dataset("small_model_train/flan", split=args.split)
dataset = get_subset(dataset)
ds = dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
ds = ds.map(_tokenize).with_format('torch', device=device)


# generate data for training
for i, elem in enumerate(tqdm(ds)):
    prompt = elem['question'].to(device)
    attention_mask = elem['attention_mask'].to(device)
    # get the correct answer first
    correct_answer = accelerator.unwrap_model(model).diffusion_generate(
        prompt,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        steps=args.steps,
        temperature=0.,
        top_p=None,
        alg="entropy",
        alg_temp=0.1,
        top_k=None,
        block_length=args.block_length,
    )
    
    def generation_tokens_hook_func(step, x: torch.Tensor, x0: torch.Tensor, confidence: torch.Tensor, block_start, block_end, mask_index, histories):
        correct_block = correct_answer[:, block_start:block_end]
        block_confidence = torch.zeros_like(x, dtype=confidence.dtype)
        block_confidence[mask_index] = confidence
        block_confidence = block_confidence[:, block_start:block_end]
        block_token = x.clone()
        block_token[mask_index] = x0
        block_token = block_token[:, block_start:block_end]
        is_correct = (block_token == correct_block)
        histories.append({
            'step': step,
            'confidence': block_confidence.reshape(-1).to('cpu').tolist(),
            'is_correct': is_correct.reshape(-1).to('cpu').tolist(),
        })
        step += 1
        # add this check because sometimes it will never predict correct answer and stuck in this loop
        if (x[:, block_start:block_end] == block_token).all():
            x[:, block_start:block_end] = correct_block
            return step, x, histories
        x[:, block_start:block_end] = torch.where(is_correct, block_token, x[:, block_start:block_end])
        return step, x, histories
    
    output = accelerator.unwrap_model(model).diffusion_generate(
        prompt,
        attention_mask=attention_mask,
        max_new_tokens=args.gen_length,
        output_history=True,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=0.,
        top_p=None,
        alg="entropy",
        alg_temp=0.1,
        top_k=None,
        block_length=args.block_length,
        generation_tokens_hook_func=generation_tokens_hook_func
    )
    
    with open(f"small_model_train/train_data/{accelerator.process_index}-{i}.pkl", 'wb') as file:
        pickle.dump(output.history, file)