import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


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


def _model_forward(model, input_ids, attention_mask=None, output_hidden_states=False):
    """Run a model forward pass while supporting remote-code model variants."""
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if output_hidden_states:
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True

    try:
        return model(input_ids, **kwargs)
    except TypeError:
        if not output_hidden_states:
            raise

        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        previous_value = getattr(model.config, "output_hidden_states", None)
        model.config.output_hidden_states = True
        try:
            return model(input_ids, **kwargs)
        finally:
            if previous_value is None:
                try:
                    delattr(model.config, "output_hidden_states")
                except AttributeError:
                    pass
            else:
                model.config.output_hidden_states = previous_value


def _output_logits(outputs):
    """Read logits from ModelOutput, dict, or tuple-like outputs."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs[0]


def _output_hidden_states(outputs):
    """Read hidden states from ModelOutput, dict, or tuple-like outputs."""
    if hasattr(outputs, "hidden_states"):
        return outputs.hidden_states
    if isinstance(outputs, dict):
        return outputs.get("hidden_states")

    for item in reversed(outputs):
        if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
            return item
    return None


def _transformer_layer_states(hidden_states, num_layers):
    """
    Return transformer block hidden states indexed as 0..num_layers-1.

    Hugging Face models commonly return embeddings at hidden_states[0], followed
    by one tensor per transformer block. If the model returns only block states,
    use them as-is.
    """
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    hidden_states = tuple(hidden_states)
    if len(hidden_states) >= num_layers + 1:
        return hidden_states[1:num_layers + 1]
    if len(hidden_states) >= num_layers:
        return hidden_states[:num_layers]

    raise RuntimeError(
        f"Expected at least {num_layers} hidden-state tensors, got {len(hidden_states)}."
    )


def _mean_pool(hidden_state, attention_mask=None):
    """Mean pool [batch, seq_len, hidden_dim] into [batch, hidden_dim]."""
    if attention_mask is None:
        return hidden_state.mean(dim=1)

    mask = attention_mask.to(device=hidden_state.device).unsqueeze(-1)
    mask = mask.to(dtype=hidden_state.dtype)
    denom = mask.sum(dim=1).clamp_min(1)
    return (hidden_state * mask).sum(dim=1) / denom


def _capture_layer_probes(outputs, attention_mask, batch_size, layer_ids, num_layers):
    """Pool selected layer hidden states and move them to CPU float tensors."""
    layer_states = _transformer_layer_states(_output_hidden_states(outputs), num_layers)
    probes = {}
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= len(layer_states):
            raise ValueError(f"Layer {layer_id} is outside available layers 0..{len(layer_states) - 1}.")

        pooled = _mean_pool(layer_states[layer_id][:batch_size], attention_mask)
        probes[f"layer_{layer_id}"] = pooled.detach().float().cpu()
    return probes


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
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
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

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
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                logits = _output_logits(_model_forward(model, x_, attention_mask=attention_mask_))
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = _output_logits(_model_forward(model, x, attention_mask=attention_mask))

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

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


@torch.no_grad()
def generate_with_layer_probes(model, prompt, attention_mask=None, steps=64, gen_length=64, block_length=64,
                               temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
                               logits_eos_inf=False, confidence_eos_eot_inf=False, probe_step=10,
                               layer_ids=None):
    """
    Generate with LLaDA and extract pooled hidden states at a denoising step.

    Args:
        model: Mask predictor.
        prompt: Tokenized prompts of shape [batch, prompt_len].
        attention_mask: Optional prompt attention mask of shape [batch, prompt_len].
        steps: Total denoising steps. For layer probing, use 64.
        gen_length: Number of generated mask tokens. Defaults to 64 for a single
            64-step denoising block.
        block_length: Semi-autoregressive block size. Defaults to gen_length.
        probe_step: One-based global denoising step to capture, e.g. 10 captures
            the tenth model forward pass out of 64.
        layer_ids: Transformer layer ids to capture. Defaults to all model layers.

    Returns:
        (x, probes) where x is the generated token tensor and probes is:
        {"layer_0": tensor[num_prompts, hidden_dim], ...}
    """
    if probe_step < 1:
        raise ValueError("probe_step is one-based and must be >= 1.")

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device),
            ],
            dim=-1,
        )
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    num_layers = getattr(model.config, "num_hidden_layers", 32)
    if layer_ids is None:
        layer_ids = list(range(num_layers))
    else:
        layer_ids = list(layer_ids)

    probes = None
    global_step = 0

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            global_step += 1
            capture_step = global_step == probe_step
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    capture_attention_mask = attention_mask
                else:
                    attention_mask_ = None
                    capture_attention_mask = None

                outputs = _model_forward(
                    model,
                    x_,
                    attention_mask=attention_mask_,
                    output_hidden_states=capture_step,
                )
                logits = _output_logits(outputs)
                if capture_step:
                    probes = _capture_layer_probes(
                        outputs,
                        capture_attention_mask,
                        prompt.shape[0],
                        layer_ids,
                        num_layers,
                    )
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                outputs = _model_forward(
                    model,
                    x,
                    attention_mask=attention_mask,
                    output_hidden_states=capture_step,
                )
                logits = _output_logits(outputs)
                if capture_step:
                    probes = _capture_layer_probes(
                        outputs,
                        attention_mask,
                        prompt.shape[0],
                        layer_ids,
                        num_layers,
                    )

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
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

    if probes is None:
        raise ValueError(f"probe_step={probe_step} was not reached; total denoising steps={global_step}.")

    return x, probes


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
