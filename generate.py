# This code is adapted from: https://github.com/ML-GSAI/LLaDA and https://github.com/NVlabs/Fast-dLLM
import torch
import numpy as np
import torch.nn.functional as F
from model.modeling_llada import LLaDAModelLM
from model.small_model import LogisticRegression

from transformers import AutoTokenizer


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate_predict_eot(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
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

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break
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
            
            # before x0 and x0_p get masked
            end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            if (end_token_conf > -np.inf).any():
                # If the end token is present, we find its position and truncate the sequence.
                # This is to ensure that the generation stops at the end token.
                position_of_end_token = end_token_conf.argmax(dim=-1)
                # print(f"Position of end token: {position_of_end_token}/{total_length}")
                x = x[:, :position_of_end_token + 1]

    return x


@ torch.no_grad()
def generate_learn2parallel(model: LLaDAModelLM, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
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

    # results = []
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        # prev_data = None
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
            
            # small model predict
            block_confidence = x0_p[:, block_start:block_end]
            block_logists = small_model(block_confidence.to(dtype=torch.float32))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
            transfer_index = torch.where(mask_index, transfer_index, False)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # unmask the highest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            i += 1

    return x


@ torch.no_grad()
def generate_l2p_eot(model: LLaDAModelLM, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
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

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
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
            
            # small model predict
            block_confidence = x0_p[:, block_start:block_end]
            block_logists = small_model(block_confidence.to(dtype=torch.float32))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
            transfer_index = torch.where(mask_index, transfer_index, False)

            # Get small_model's 'endoftext' prediction
            small_model_eot = ((x0 == 126081) & transfer_index)
            # before x0 and x0_p get masked
            end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)
            
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            for j in range(confidence.shape[0]):
                # unmask the highest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]

            # ignore any token after 'endoftext' token
            if small_model_eot.any():
                position_of_end_token = small_model_eot.to(torch.int8).argmax(dim=-1)[0]
                x = x[:, :position_of_end_token + 1]
            elif (end_token_conf > -np.inf).any():
                # If the end token is present, we find its position and truncate the sequence.
                # This is to ensure that the generation stops at the end token.
                position_of_end_token = end_token_conf.argmax(dim=-1)
                x = x[:, :position_of_end_token + 1]
            i += 1

    return x


@ torch.no_grad()
def generate(model: LLaDAModelLM, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
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

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
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

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


#------Beginning of code adopted from Fast-dLLM-----#
@ torch.no_grad()
def generate_with_prefix_cache(model, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336):
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
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        if current_block_start >= x.shape[1]:
            break
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index, position_of_end_token = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0], small_model, accept_thres, current_block_start, current_block_end)
        x[transfer_index] = x0[transfer_index]
        if position_of_end_token is not None:
            x = x[:, :position_of_end_token + 1]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            x0, transfer_index, position_of_end_token = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i], small_model, accept_thres, block_start=0, block_end=current_block_end-current_block_start)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            if position_of_end_token is not None:
                x = x[:, :current_block_start + position_of_end_token + 1]
            
            i += 1

    return x


@ torch.no_grad()
def generate_with_dual_cache(model, small_model: LogisticRegression, accept_thres: float, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336):
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
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        if current_block_start >= x.shape[1]:
            break
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index, position_of_end_token = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0], small_model, accept_thres, current_block_start, current_block_end)
        x[transfer_index] = x0[transfer_index]
        if position_of_end_token is not None:
            x = x[:, :position_of_end_token + 1]

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            x0, transfer_index, position_of_end_token = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:current_block_end], num_transfer_tokens[:, i], small_model, accept_thres)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if position_of_end_token is not None:
                x = x[:, :current_block_start + position_of_end_token + 1]
                replace_position[:, current_block_start + position_of_end_token + 1:] = 0
            i += 1

    return x


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, small_model: LogisticRegression, accept_thres: float, block_start = None, block_end = None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    # small model predict
    if block_start is not None and block_end is not None:
        block_confidence = x0_p[:, block_start:block_end]
        block_logists = small_model(block_confidence.to(dtype=torch.float32))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        transfer_index[:, block_start:block_end] = (torch.sigmoid(block_logists) > accept_thres)
    else:
        block_confidence = x0_p
        block_logists = small_model(block_confidence.to(dtype=torch.float32))
        transfer_index = (torch.sigmoid(block_logists) > accept_thres)
    transfer_index = torch.where(mask_index, transfer_index, False)

    # Get small_model's 'endoftext' prediction
    small_model_eot = ((x0 == 126081) & transfer_index)
    # before x0 and x0_p get masked
    end_token_conf = torch.where((x0 == 126081) & mask_index, x0_p, -np.inf)
    # ignore any token after 'endoftext' token
    position_of_end_token = None
    if small_model_eot.any():
        position_of_end_token = small_model_eot.to(torch.int8).argmax(dim=-1)[0]
    elif (end_token_conf > -np.inf).any():
        # If the end token is present, we find its position and truncate the sequence.
        # This is to ensure that the generation stops at the end token.
        position_of_end_token = end_token_conf.argmax(dim=-1)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    for j in range(confidence.shape[0]):
        # unmask the highest
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
    return x0, transfer_index, position_of_end_token
#----Beginning of code adopted from Fast-dLLM----#


def main():
    device = 'cuda'

    small_model = LogisticRegression(32)
    small_model.load_state_dict(torch.load('small_model_train/layer_2_flan.pth'))
    small_model.to(device)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours? Give me the number only"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    import time
    start = time.time()
    out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_predict_eot(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_learn2parallel(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_l2p_eot(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_with_prefix_cache(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    # out = generate_with_dual_cache(model, small_model, 0.96, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    print(f'{time.time() - start}')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()