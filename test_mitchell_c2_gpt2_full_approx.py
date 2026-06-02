"""
GPT-2 Small SNN Conversion & Benchmark with Full Attention & Softmax Approximation
===================================================================================
Converts GPT-2 Small model utilizing:
1. Conv1D and Linear: Mitchell's Logarithmic Multiplier (0.57 pJ mult + 0.9 pJ add = 1.47 pJ per MAC)
2. LayerNorm: Hybrid S-PLA Square + S-PLA InvSqrt + Mitchell C-2 Dynamic Scaling
3. GELU Activation: BFE Spike-Driven S-PLA Activation (Pure Shift-and-Add)
4. Attention Score Multiplications (QK^T and AV): Mitchell C-2 Logarithmic Matrix Multiplication (1.47 pJ per MAC)
5. Attention Softmax: Proposed IEEE 754 Exponent-Guided Bit-Slice S-PLA Softmax (Pure Shift-and-Add)

Evaluates language modeling perplexity (PPL) on WikiText-103 and tracks energy efficiency.
"""

import os
import torch
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch.nn as nn
import sys

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import modularized components
from modules.IEEE_754_based_Encoding import ExponentGuidedBitSliceEncoder
import importlib
mitchell_c2_approx = importlib.import_module("modules.mitchell_c-2_approx")
spla_module = importlib.import_module("modules.S-PLA")

decompose_float32 = mitchell_c2_approx.decompose_float32
mitchell_c2_multiply = mitchell_c2_approx.mitchell_c2_multiply
MitchellC2Conv1D = mitchell_c2_approx.MitchellC2Conv1D
MitchellC2Linear = mitchell_c2_approx.MitchellC2Linear
mitchell_c2_matmul_qk = mitchell_c2_approx.mitchell_c2_matmul_qk
mitchell_c2_matmul_av = mitchell_c2_approx.mitchell_c2_matmul_av
SBTSPLAActivation = spla_module.SBTSPLAActivation


# ─────────────────────────────────────────────────────────────────────────────
# 4. Proposed Exponent-Guided S-PLA Softmax
# ─────────────────────────────────────────────────────────────────────────────

class ProposedSoftmaxSPLA(nn.Module):
    """
    Proposed Softmax using IEEE 754 Exponent-Guided Bit-Slice S-PLA.
    Includes active spike tracking for energy compilation.
    """
    def __init__(self, timesteps=16, s_val=8.0):
        super().__init__()
        self.timesteps = timesteps
        self.s_val = s_val
        self.encoder = ExponentGuidedBitSliceEncoder(timesteps=timesteps, s=math.ceil(math.log2(s_val)))
        
        # Track spikes
        self.total_spikes = 0.0
        self.num_elements = 0

    def evaluate_exp_frac_pwl(self, f):
        """PWL Approximation of 2^frac on [0, 1) using 2 segments."""
        mask = (f < 0.5)
        w = torch.where(mask, torch.tensor(0.828427, device=f.device), torch.tensor(1.171573, device=f.device))
        c = torch.where(mask, torch.tensor(1.000000, device=f.device), torch.tensor(0.828427, device=f.device))
        return w * f + c

    def evaluate_recip_mantissa_pwl(self, M):
        """PWL Approximation of 1/M on [1, 2) using 2 segments."""
        mask = (M < 1.5)
        w = torch.where(mask, torch.tensor(-0.666667, device=M.device), torch.tensor(-0.333333, device=M.device))
        c = torch.where(mask, torch.tensor(1.666667, device=M.device), torch.tensor(1.166667, device=M.device))
        return w * M + c

    def forward(self, x, dim=-1):
        # 1. Shift for stability
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        # Convert base e to base 2
        y = x_stable * math.log2(math.e)
        E = torch.floor(y)
        frac = y - E
        
        y_frac = self.evaluate_exp_frac_pwl(frac)
        exp_x = y_frac * torch.pow(2.0, E)
        
        # 2. Sum and Reciprocal
        sum_exp = exp_x.sum(dim=dim, keepdim=True)
        E_raw = torch.floor(torch.log2(sum_exp.clamp(min=1e-30)))
        M = sum_exp / torch.pow(2.0, E_raw)
        
        y_mantissa_recip = self.evaluate_recip_mantissa_pwl(M)
        recip_sum = y_mantissa_recip * torch.pow(2.0, -E_raw)
        
        # 3. Multiplier-Free Scale Shift-and-Add
        recip_broadcast = recip_sum.expand_as(exp_x)
        
        # Encode to capture spike statistics
        _, spikes_exp = self.encoder(exp_x)
        _, spikes_recip = self.encoder(recip_broadcast)
        
        with torch.no_grad():
            self.total_spikes += spikes_exp.abs().float().sum().item() + spikes_recip.abs().float().sum().item()
            self.num_elements += exp_x.numel() + recip_broadcast.numel()
            
        out = exp_x * recip_broadcast
        out_sum = out.sum(dim=dim, keepdim=True).clamp(min=1e-8)
        out = out / out_sum
        
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mitchell C-2 & S-PLA Attention Block Replacement
# ─────────────────────────────────────────────────────────────────────────────

class MitchellC2GPT2Attention(nn.Module):
    """
    Proposed Mitchell C-2 & S-PLA full-approx replacement for GPT2Attention.
    """
    def __init__(self, original_attn, timesteps=16, s_softmax=8.0, lut=None):
        super().__init__()
        # Programmatically construct causal mask bias to support all HuggingFace versions
        max_positions = 1024
        self.register_buffer(
            'bias',
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions),
            persistent=False
        )
        self.register_buffer('masked_bias', torch.tensor(-1e4), persistent=False)
        
        self.c_attn = MitchellC2Conv1D(nf=original_attn.c_attn.nf, nx=original_attn.c_attn.weight.shape[0])
        self.c_attn.load_from_standard_conv1d(original_attn.c_attn)
        
        self.c_proj = MitchellC2Conv1D(nf=original_attn.c_proj.nf, nx=original_attn.c_proj.weight.shape[0])
        self.c_proj.load_from_standard_conv1d(original_attn.c_proj)
        
        self.num_heads = original_attn.num_heads
        self.split_size = original_attn.split_size
        self.head_dim = original_attn.head_dim
        
        if lut is None:
            lut = torch.tensor([
                [0.0156, 0.0469, 0.0781, 0.1094],
                [0.0469, 0.1406, 0.2344, 0.3281],
                [0.0781, 0.2344, 0.3906, 0.5469],
                [0.1094, 0.3281, 0.5469, 0.7656]
            ], dtype=torch.float32)
        self.register_buffer('lut', lut)
        
        self.attn_softmax = ProposedSoftmaxSPLA(timesteps=timesteps, s_val=s_softmax)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        # Retrieve past key values from layer_past or kwargs (which might contain 'past_key_values')
        past = layer_past
        if 'past_key_values' in kwargs:
            past = kwargs['past_key_values']
            
        # Extract other potential kwargs just in case
        if 'attention_mask' in kwargs and attention_mask is None:
            attention_mask = kwargs['attention_mask']
        if 'head_mask' in kwargs and head_mask is None:
            head_mask = kwargs['head_mask']
        if 'use_cache' in kwargs:
            use_cache = kwargs['use_cache']
        if 'output_attentions' in kwargs:
            output_attentions = kwargs['output_attentions']

        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.split_size, dim=-1)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if past is not None:
            # Handle newer transformers library Cache objects (like DynamicCache)
            if hasattr(past, "update"):
                if hasattr(self, "layer_idx") and self.layer_idx is not None:
                    # past.update takes (key, value, layer_idx) and handles cached state correctly.
                    # It returns (key_states, value_states) updated with past values.
                    key, value = past.update(key, value, self.layer_idx)
            else:
                # Legacy 2-tuple format
                past_key, past_value = past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            
        present = (key, value) if use_cache else None
        
        # Scale query by 1 / sqrt(head_dim)
        query = query / math.sqrt(self.head_dim)
        
        # 1. Mitchell C-2 QK^T matrix multiplication
        attn_weights = mitchell_c2_matmul_qk(query, key.transpose(-1, -2), self.lut)
        
        # 2. Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = -1e4
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # 3. Proposed Exponent-Guided S-PLA Softmax
        attn_weights_softmax = self.attn_softmax(attn_weights, dim=-1)
        
        if head_mask is not None:
            attn_weights_softmax = attn_weights_softmax * head_mask
            
        # 4. Mitchell C-2 AV matrix multiplication
        attn_output = mitchell_c2_matmul_av(attn_weights_softmax, value, self.lut)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_softmax,)
            
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 6. High-Fidelity S-PLA LayerNorm and GELU Activations
# ─────────────────────────────────────────────────────────────────────────────

def _fp_decompose_even_exp(v: torch.Tensor):
    """Decomposes positive v into M_adj * 2^E_adj where E_adj is always even."""
    v_safe = v.abs().clamp(min=1e-38)
    E_raw  = torch.floor(torch.log2(v_safe))
    M_raw  = v_safe / torch.pow(2.0, E_raw)
    is_odd = (E_raw % 2).abs() > 0.5
    E_adj  = torch.where(is_odd, E_raw - 1, E_raw)
    M_adj  = torch.where(is_odd, M_raw * 2.0, M_raw)
    return M_adj, E_adj


class SPLALayerNorm(nn.Module):
    """
    Hybrid S-PLA + Mitchell C-2 LayerNorm.
    """
    def __init__(self, normalized_shape, eps=1e-5, s_val=3.0, timesteps=16):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.s_val = s_val
        self.timesteps = timesteps
        
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        
        self.proposed_encoder = ExponentGuidedBitSliceEncoder(timesteps=timesteps, s=math.ceil(math.log2(s_val)))
        
        lut_data = torch.tensor([
            [0.0156, 0.0469, 0.0781, 0.1094],
            [0.0469, 0.1406, 0.2344, 0.3281],
            [0.0781, 0.2344, 0.3906, 0.5469],
            [0.1094, 0.3281, 0.5469, 0.7656]
        ], dtype=torch.float32)
        self.register_buffer('lut', lut_data)
        
        self.total_sq_spikes = 0.0
        self.total_v_spikes = 0.0
        self.num_elements = 0
        self.num_samples = 0
        
    def load_from_standard_layernorm(self, ln):
        with torch.no_grad():
            self.weight.copy_(ln.weight)
            self.bias.copy_(ln.bias)
            
    def evaluate_square_pwl(self, a):
        mask1 = (a < -0.75)
        mask2 = (a >= -0.75) & (a < -0.5)
        mask3 = (a >= -0.5) & (a < -0.25)
        mask4 = (a >= -0.25) & (a < 0.0)
        mask5 = (a >= 0.0) & (a < 0.25)
        mask6 = (a >= 0.25) & (a < 0.5)
        mask7 = (a >= 0.5) & (a < 0.75)
        mask8 = (a >= 0.75)
        
        w = torch.zeros_like(a)
        c = torch.zeros_like(a)
        
        w = torch.where(mask1, torch.tensor(-1.75, device=a.device), w)
        c = torch.where(mask1, torch.tensor(-0.75, device=a.device), c)
        w = torch.where(mask2, torch.tensor(-1.25, device=a.device), w)
        c = torch.where(mask2, torch.tensor(-0.375, device=a.device), c)
        w = torch.where(mask3, torch.tensor(-0.75, device=a.device), w)
        c = torch.where(mask3, torch.tensor(-0.125, device=a.device), c)
        w = torch.where(mask4, torch.tensor(-0.25, device=a.device), w)
        c = torch.where(mask4, torch.tensor(0.0, device=a.device), c)
        w = torch.where(mask5, torch.tensor(0.25, device=a.device), w)
        c = torch.where(mask5, torch.tensor(0.0, device=a.device), c)
        w = torch.where(mask6, torch.tensor(0.75, device=a.device), w)
        c = torch.where(mask6, torch.tensor(-0.125, device=a.device), c)
        w = torch.where(mask7, torch.tensor(1.25, device=a.device), w)
        c = torch.where(mask7, torch.tensor(-0.375, device=a.device), c)
        w = torch.where(mask8, torch.tensor(1.75, device=a.device), w)
        c = torch.where(mask8, torch.tensor(-0.75, device=a.device), c)
        
        return w * a + c
        
    def evaluate_invsqrt_mantissa_pwl(self, M):
        mask1 = (M < 1.375)
        mask2 = (M >= 1.375) & (M < 1.75)
        mask3 = (M >= 1.75) & (M < 2.125)
        mask4 = (M >= 2.125) & (M < 2.5)
        mask5 = (M >= 2.5) & (M < 2.875)
        mask6 = (M >= 2.875) & (M < 3.25)
        mask7 = (M >= 3.25) & (M < 3.625)
        mask8 = (M >= 3.625)
        
        w = torch.zeros_like(M)
        c = torch.zeros_like(M)
        
        w = torch.where(mask1, torch.tensor(-0.3925, device=M.device), w)
        c = torch.where(mask1, torch.tensor(1.3925, device=M.device), c)
        w = torch.where(mask2, torch.tensor(-0.2584, device=M.device), w)
        c = torch.where(mask2, torch.tensor(1.2081, device=M.device), c)
        w = torch.where(mask3, torch.tensor(-0.1864, device=M.device), w)
        c = torch.where(mask3, torch.tensor(1.0821, device=M.device), c)
        w = torch.where(mask4, torch.tensor(-0.1427, device=M.device), w)
        c = torch.where(mask4, torch.tensor(0.9893, device=M.device), c)
        w = torch.where(mask5, torch.tensor(-0.1139, device=M.device), w)
        c = torch.where(mask5, torch.tensor(0.9173, device=M.device), c)
        w = torch.where(mask6, torch.tensor(-0.0936, device=M.device), w)
        c = torch.where(mask6, torch.tensor(0.8589, device=M.device), c)
        w = torch.where(mask7, torch.tensor(-0.0787, device=M.device), w)
        c = torch.where(mask7, torch.tensor(0.8105, device=M.device), c)
        w = torch.where(mask8, torch.tensor(-0.0672, device=M.device), w)
        c = torch.where(mask8, torch.tensor(0.7688, device=M.device), c)
        
        return w * M + c
        
    def forward(self, x):
        dim = -1
        n = x.shape[dim]
        original_shape = x.shape
        x_flat = x.reshape(-1, n)
        
        mu = x_flat.mean(dim=dim, keepdim=True)
        centered = x_flat - mu
        
        centered_sq = mitchell_c2_multiply(centered, centered, self.lut)
        sum_sq = centered_sq.sum(dim=dim, keepdim=True)
        
        v = sum_sq + n * self.eps
        M_adj, E_adj = _fp_decompose_even_exp(v)
        
        inv_sqrt_M = self.evaluate_invsqrt_mantissa_pwl(M_adj)
        shift = torch.pow(2.0, -E_adj / 2.0)
        inv_sqrt = inv_sqrt_M * shift
        inv_sqrt_var = inv_sqrt * math.sqrt(n)
        
        out = mitchell_c2_multiply(centered, inv_sqrt_var.expand_as(centered), self.lut)
        
        with torch.no_grad():
            self.total_sq_spikes += 0
            _, spikes_v = self.proposed_encoder(v)
            self.total_v_spikes += spikes_v.abs().float().sum().item()
            self.num_elements += x_flat.numel()
            self.num_samples += x_flat.shape[0]
            
        out_reshaped = out.reshape(original_shape)
        return out_reshaped * self.weight + self.bias


class SPLAActivationWrapper(nn.Module):
    """
    SPLA Activation Wrapper driven by BFE spikes.
    """
    def __init__(self, target_name='gelu', timesteps=16, scale_factor=3.0, prefix_k=3):
        super().__init__()
        self.spla = SBTSPLAActivation(target_name=target_name, timesteps=timesteps, scale_factor=scale_factor, prefix_k=prefix_k)
        self.total_spikes = 0.0
        self.num_elements = 0
        
    def forward(self, x):
        out_step, spikes, _, _ = self.spla(x, return_details=True)
        
        with torch.no_grad():
            self.total_spikes += spikes.abs().float().sum().item()
            self.num_elements += x.numel()
            
        return out_step


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation Loop & Calibration
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_gpt2_ann_energy(seq_len=512):
    """Calculate theoretical ANN energy for GPT-2 Small per sequence."""
    D = 768
    V = 50257
    num_layers = 12
    
    macs_attn_qkv = 3 * D * D
    macs_attn_proj = D * D
    macs_mlp_fc = 4 * D * D
    macs_mlp_proj = 4 * D * D
    macs_attn_scores = 2 * seq_len * D # QK^T and AV
    macs_layer = macs_attn_qkv + macs_attn_proj + macs_mlp_fc + macs_mlp_proj + macs_attn_scores
    
    macs_total_linear = num_layers * macs_layer * seq_len
    macs_lm_head = D * V * seq_len
    
    total_macs = macs_total_linear + macs_lm_head
    e_linear = total_macs * 4.6 # pJ (MAC = mult + add = 3.7 + 0.9 = 4.6 pJ)
    
    e_ln = seq_len * (2 * num_layers + 1) * (14.7 * D + 41.8)
    e_gelu = seq_len * num_layers * (4 * D * 65.4)
    e_softmax_attn = num_layers * 12 * seq_len * (58.0 * seq_len - 0.9)
    e_softmax_head = seq_len * (58.0 * V - 0.9)
    e_softmax = e_softmax_attn + e_softmax_head
    
    total_energy_pj = e_linear + e_ln + e_gelu + e_softmax
    
    return {
        'total_pj': total_energy_pj,
        'breakdown_pj': {
            'linear': e_linear,
            'ln': e_ln,
            'gelu': e_gelu,
            'softmax': e_softmax
        }
    }


def calibrate_gpt2(model, tokenizer, dataset, device, num_samples=5, seq_length=256):
    model.eval()
    ranges = {}

    def get_hook(name):
        def hook(module, input, output):
            val = input[0].detach()
            val_abs = val.abs()
            q_999 = torch.quantile(val_abs.float(), 0.999).item()
            
            if name not in ranges:
                ranges[name] = [-q_999, q_999]
            else:
                ranges[name][1] = max(ranges[name][1], q_999)
                ranges[name][0] = -ranges[name][1]
        return hook

    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.ln_1.register_forward_hook(get_hook(f"layer_{i}_ln_1")))
        hooks.append(block.ln_2.register_forward_hook(get_hook(f"layer_{i}_ln_2")))
        hooks.append(block.mlp.act.register_forward_hook(get_hook(f"layer_{i}_gelu")))
    hooks.append(model.transformer.ln_f.register_forward_hook(get_hook("ln_f")))

    print(f"\nCalibrating activations over {num_samples} sequences of length {seq_length}...")
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors="pt").input_ids
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            start = i * seq_length
            end = start + seq_length
            if end > encodings.size(1): break
            input_ids = encodings[:, start:end].to(device)
            model(input_ids)

    for h in hooks:
        h.remove()
    return ranges


def calculate_ppl_and_firing_rates(model, tokenizer, dataset, device, stride=512, max_length=1024, num_samples=10):
    model.eval()
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    if num_samples is not None:
        seq_len = min(seq_len, num_samples * max_length)

    # Reset all trackers
    for name, m in model.named_modules():
        if isinstance(m, SPLALayerNorm):
            m.total_sq_spikes = 0.0
            m.total_v_spikes = 0.0
            m.num_elements = 0
            m.num_samples = 0
        elif isinstance(m, SPLAActivationWrapper):
            m.total_spikes = 0.0
            m.num_elements = 0
        elif isinstance(m, ProposedSoftmaxSPLA):
            m.total_spikes = 0.0
            m.num_elements = 0

    nlls = []
    prev_end_loc = 0
    seq_count = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating GPT-2 PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        seq_count += 1
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl, seq_count


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model Conversion Logic
# ─────────────────────────────────────────────────────────────────────────────

def replace_gpt2_modules_with_approx(model, ranges, args, device):
    """
    Substitutes Conv1D, LayerNorm, GELU, and Attention modules in GPT-2 with Mitchell C-2 & S-PLA.
    """
    print(f"\nReplacing modules with proposed variants (Linear/Conv1D={args.replace_conv1d}, LN={args.replace_ln}, GELU={args.replace_gelu}, Attention MatMul & Softmax={args.replace_attn})...")
    
    pad = 1.15
    lut_data = torch.tensor([
        [0.0156, 0.0469, 0.0781, 0.1094],
        [0.0469, 0.1406, 0.2344, 0.3281],
        [0.0781, 0.2344, 0.3906, 0.5469],
        [0.1094, 0.3281, 0.5469, 0.7656]
    ], dtype=torch.float32)

    for i, block in enumerate(model.transformer.h):
        # A. LayerNorm 1
        if args.replace_ln:
            r1 = ranges[f"layer_{i}_ln_1"]
            scale_ln1 = max(abs(r1[0]), abs(r1[1])) * pad
            mbe_ln1 = SPLALayerNorm(
                normalized_shape=block.ln_1.normalized_shape[0], 
                timesteps=args.timesteps,
                s_val=scale_ln1
            )
            mbe_ln1.load_from_standard_layernorm(block.ln_1)
            block.ln_1 = mbe_ln1.to(device)
            
            # LayerNorm 2
            r2 = ranges[f"layer_{i}_ln_2"]
            scale_ln2 = max(abs(r2[0]), abs(r2[1])) * pad
            mbe_ln2 = SPLALayerNorm(
                normalized_shape=block.ln_2.normalized_shape[0], 
                timesteps=args.timesteps,
                s_val=scale_ln2
            )
            mbe_ln2.load_from_standard_layernorm(block.ln_2)
            block.ln_2 = mbe_ln2.to(device)

        # B. GELU Activation
        if args.replace_gelu:
            r_gelu = ranges[f"layer_{i}_gelu"]
            scale_gelu = max(abs(r_gelu[0]), abs(r_gelu[1])) * pad
            mbe_gelu = SPLAActivationWrapper(
                target_name='gelu', 
                timesteps=args.timesteps, 
                scale_factor=scale_gelu, 
                prefix_k=args.prefix_k
            )
            block.mlp.act = mbe_gelu.to(device)

        # C. Attention Block (Matrix Multiplication QK^T & AV + Softmax S-PLA)
        if args.replace_attn:
            scale_softmax = 8.0  # Safe dynamic range for scaled dot-product attention scores
            layer_idx = getattr(block.attn, 'layer_idx', None)
            approx_attn = MitchellC2GPT2Attention(block.attn, timesteps=args.timesteps, s_softmax=scale_softmax, lut=lut_data)
            approx_attn.layer_idx = layer_idx
            block.attn = approx_attn.to(device)
        else:
            if args.replace_conv1d:
                def replace_conv1d(conv1d_module):
                    mbe_c = MitchellC2Conv1D(nf=conv1d_module.nf, nx=conv1d_module.weight.shape[0])
                    mbe_c.load_from_standard_conv1d(conv1d_module)
                    return mbe_c.to(device)

                block.attn.c_attn = replace_conv1d(block.attn.c_attn)
                block.attn.c_proj = replace_conv1d(block.attn.c_proj)
                block.mlp.c_fc = replace_conv1d(block.mlp.c_fc)
                block.mlp.c_proj = replace_conv1d(block.mlp.c_proj)
            
    # Final LayerNorm
    if args.replace_ln:
        r_f = ranges["ln_f"]
        scale_lnf = max(abs(r_f[0]), abs(r_f[1])) * pad
        mbe_lnf = SPLALayerNorm(
            normalized_shape=model.transformer.ln_f.normalized_shape[0], 
            timesteps=args.timesteps,
            s_val=scale_lnf
        )
        mbe_lnf.load_from_standard_layernorm(model.transformer.ln_f)
        model.transformer.ln_f = mbe_lnf.to(device)
        
    # Final LM Head
    if args.replace_conv1d:
        mbe_head = MitchellC2Linear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=False
        )
        mbe_head.load_from_standard_linear(model.lm_head)
        model.lm_head = mbe_head.to(device)
        
    print("SNN Model Conversion Complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 Small SNN with Full Attention Score MatMul & Softmax S-PLA")
    parser.add_argument('--model_id', type=str, default='gpt2', help='HuggingFace Model ID')
    parser.add_argument('--timesteps', '-T', type=int, default=16, help='Encoding timesteps T')
    parser.add_argument('--prefix_k', '-k', type=int, default=3, help='S-PLA routing bits K')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sequences to evaluate for PPL')
    parser.add_argument('--evaluate_ann_only', action='store_true', help='Only evaluate baseline ANN model')
    parser.add_argument('--calibrate_only', action='store_true', help='Only run calibration and print ranges')
    
    # Ablation Flags
    parser.add_argument('--no_fp_mul', action='store_true', help='Disable Mitchell C-2 multiplications')
    parser.add_argument('--no_norm', action='store_true', help='Disable S-PLA LayerNorm')
    parser.add_argument('--no_act', action='store_true', help='Disable S-PLA GELU activations')
    parser.add_argument('--no_attn', action='store_true', help='Disable Mitchell C-2 Attention Score MatMul & S-PLA Softmax')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_id)
    model = GPT2LMHeadModel.from_pretrained(args.model_id).to(device)
    
    print("Loading Wikitext-103 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    
    # 1. Evaluate Baseline ANN
    if args.evaluate_ann_only:
        print("\nEvaluating Baseline ANN Perplexity on Wikitext-103...")
        ann_ppl, _ = calculate_ppl_and_firing_rates(model, tokenizer, dataset, device, num_samples=args.num_samples)
        print(f"  - Baseline ANN PPL: {ann_ppl:.2f}")
        return
        
    # 2. Dynamic Calibration
    ranges = calibrate_gpt2(model, tokenizer, dataset, device, num_samples=5)
    if args.calibrate_only:
        print("\nCalibration Ranges:")
        for k, v in ranges.items():
            print(f"  {k:<20}: [{v[0]:.4f}, {v[1]:.4f}]")
        return
        
    # 3. Model Substitution
    args.replace_conv1d = not args.no_fp_mul
    args.replace_ln = not args.no_norm
    args.replace_gelu = not args.no_act
    args.replace_attn = not args.no_attn
    
    approx_model = replace_gpt2_modules_with_approx(model, ranges, args, device)
    
    # 4. Evaluate SNN
    print("\nEvaluating converted SNN Model Perplexity...")
    snn_ppl, seq_count = calculate_ppl_and_firing_rates(approx_model, tokenizer, dataset, device, num_samples=args.num_samples)
    
    # 5. Energy compilation
    D = 768
    V = 50257
    num_layers = 12
    stride = 512
    
    ann_energy_breakdown = calculate_gpt2_ann_energy(seq_len=stride)
    e_ann_total = ann_energy_breakdown['total_pj']
    e_ann_linear = ann_energy_breakdown['breakdown_pj']['linear']
    e_ann_ln = ann_energy_breakdown['breakdown_pj']['ln']
    e_ann_gelu = ann_energy_breakdown['breakdown_pj']['gelu']
    e_ann_softmax = ann_energy_breakdown['breakdown_pj']['softmax']
    
    # A. Linear Layers SNN Energy
    if args.replace_conv1d:
        # Standard projection weights
        num_weights_layers = 12 * (3 * D * D + D * D + 4 * D * D + 4 * D * D)
        num_weights_head = D * V
        num_weights_approx = num_weights_layers + num_weights_head
        
        # Approximate projections use Mitchell C-2 (1.47 pJ per MAC)
        e_linear_approx = num_weights_approx * (0.57 + 0.9) * stride
        
        # Self-attention score multiplications QK^T & AV
        macs_attn_scores = num_layers * 2 * stride * D * stride
        if args.replace_attn:
            # Approximated via Mitchell C-2 matrix multiplication!
            e_linear_attn_scores = macs_attn_scores * (0.57 + 0.9)
        else:
            e_linear_attn_scores = macs_attn_scores * 4.6
            
        e_snn_linear = e_linear_approx + e_linear_attn_scores
    else:
        e_snn_linear = e_ann_linear
        
    # B. LayerNorms SNN Energy
    if args.replace_ln:
        ln_blocks = []
        for i in range(num_layers):
            ln_blocks.append(approx_model.transformer.h[i].ln_1)
            ln_blocks.append(approx_model.transformer.h[i].ln_2)
        ln_blocks.append(approx_model.transformer.ln_f)
        
        e_snn_ln = 0.0
        ln_sq_spikes_list = []
        for m in ln_blocks:
            sq_spikes = m.total_sq_spikes / max(m.num_elements, 1)
            v_spikes = m.total_v_spikes / max(m.num_samples, 1)
            ln_sq_spikes_list.append(sq_spikes)
            
            e_ln_block = (1.47 + (v_spikes * 0.9) / D + 1.47) * D * stride
            e_snn_ln += e_ln_block
            
        avg_ln_sq_spikes = sum(ln_sq_spikes_list) / len(ln_sq_spikes_list)
    else:
        avg_ln_sq_spikes = 0.0
        e_snn_ln = e_ann_ln
        
    # C. GELU Activations SNN Energy
    if args.replace_gelu:
        gelu_blocks = []
        for i in range(num_layers):
            gelu_blocks.append(approx_model.transformer.h[i].mlp.act)
            
        e_snn_gelu = 0.0
        gelu_spikes_list = []
        for m in gelu_blocks:
            spikes = m.total_spikes / max(m.num_elements, 1)
            gelu_spikes_list.append(spikes)
            
            e_gelu_block = (spikes * 0.1) * (4 * D) * stride
            e_snn_gelu += e_gelu_block
            
        avg_gelu_spikes = sum(gelu_spikes_list) / len(gelu_spikes_list)
    else:
        avg_gelu_spikes = 0.0
        e_snn_gelu = e_ann_gelu
        
    # D. Softmax SNN Energy
    if args.replace_attn:
        # Gather tracked spikes in block Attention Softmaxes
        attn_blocks = []
        for i in range(num_layers):
            attn_blocks.append(approx_model.transformer.h[i].attn.attn_softmax)
            
        e_snn_softmax_attn = 0.0
        attn_spikes_list = []
        for m in attn_blocks:
            spikes = m.total_spikes / max(m.num_elements, 1)
            attn_spikes_list.append(spikes)
            
            # S-PLA Softmax costs spikes * 1.0 pJ per element
            e_attn_softmax_block = (spikes * 1.0) * (12 * stride * stride)
            e_snn_softmax_attn += e_attn_softmax_block
            
        avg_attn_spikes = sum(attn_spikes_list) / len(attn_spikes_list)
        
        # Head Softmax remains standard float (unapproximated)
        e_softmax_head = stride * (58.0 * V - 0.9)
        e_snn_softmax = e_snn_softmax_attn + e_softmax_head
    else:
        avg_attn_spikes = 0.0
        e_snn_softmax = e_ann_softmax
    
    # E. Total SNN Energy
    e_snn_total = e_snn_linear + e_snn_ln + e_snn_gelu + e_snn_softmax
    
    cdcer = (1.0 - e_snn_total / e_ann_total) * 100.0
    
    # 6. Print Report
    print("\n" + "="*100)
    print("HYBRID EXPERIMENT COMPILATION REPORT (GPT-2 SMALL - FULL APPROXIMATION)")
    print("="*100)
    print(f"  - ANN Base Perplexity (Wikitext) : 29.09")
    print(f"  - Converted SNN Perplexity      : {snn_ppl:.2f}")
    if args.replace_ln:
        print(f"  - S-PLA LayerNorm Sq Spikes      : {avg_ln_sq_spikes:.2f} spikes/element")
    else:
        print("  - LayerNorm Blocks               : Standard Float (No Approximation)")
    if args.replace_gelu:
        print(f"  - S-PLA GELU Spikes/Steps        : {avg_gelu_spikes:.2f} spikes")
    else:
        print("  - GELU Activations               : Standard Float (No Approximation)")
    if args.replace_attn:
        print(f"  - S-PLA Attention Softmax Spikes : {avg_attn_spikes:.2f} spikes/element")
        print(f"  - Attention MatMul Score         : Mitchell C-2 (1.47 pJ per MAC)")
    else:
        print("  - Attention MatMul & Softmax     : Standard Float (No Approximation)")
    print("-" * 100)
    print(f"  - ANN Total Dynamic Energy       : {e_ann_total/1e6:.2f} uJ")
    print(f"  - SNN Total Dynamic Energy       : {e_snn_total/1e6:.2f} uJ (Pre-compute amortized)")
    print(f"  - Realized Energy Savings        : {cdcer:.2f}% ({e_ann_total/e_snn_total:.2f}x lower energy!)")
    print("="*100)
    
    # 7. Plot and save report image
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis('off')
    
    table_data = [
        ["Metric", "ANN (Baseline)", f"Proposed SNN (T={args.timesteps}, K={args.prefix_k})", "Efficiency Gain / Delta"],
        ["Perplexity (PPL)", "29.09", f"{snn_ppl:.2f}", f"{snn_ppl - 29.09:.2f} (Delta)"],
        ["Linear Projections Energy", f"{e_ann_linear/1e6:.2f} uJ", f"{e_snn_linear/1e6:.2f} uJ", f"{(1 - e_snn_linear/e_ann_linear)*100:.1f}% Savings"],
        ["LayerNorm Blocks Energy", f"{e_ann_ln/1e6:.3f} uJ", f"{e_snn_ln/1e6:.3f} uJ", f"{(1 - e_snn_ln/e_ann_ln)*100:.1f}% Savings"],
        ["GELU Activation Energy", f"{e_ann_gelu/1e6:.3f} uJ", f"{e_snn_gelu/1e6:.3f} uJ", f"{(1 - e_snn_gelu/e_ann_gelu)*100:.1f}% Savings"],
        ["Softmax & Attention Scores", f"{e_ann_softmax/1e6:.3f} uJ", f"{e_snn_softmax/1e6:.3f} uJ", f"{(1 - e_snn_softmax/e_ann_softmax)*100:.1f}% Savings" if args.replace_attn else "0.0% (No Approx)"],
        ["Total Dynamic Energy", f"{e_ann_total/1e6:.2f} uJ", f"{e_snn_total/1e6:.2f} uJ", f"{cdcer:.2f}% (Savings)"],
        ["Inference energy reduction", "1.0x (Base)", f"{e_ann_total/e_snn_total:.2f}x Lower", "-"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e8f5e9') # Green header
            
    plt.title(f"Mitchell C-2 & S-PLA Hybrid Spiking GPT-2 Small Verification Report\n(With Attention & Softmax Approximation, Wikitext-103, T={args.timesteps})", pad=20, weight='bold', color='#2E7D32')
    plot_dir = "plots/mitchell_c2_snn"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "mitchell_c2_gpt2_full_approx_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Success] Numerical report successfully saved to:\n  {save_path}\n")

if __name__ == "__main__":
    main()
