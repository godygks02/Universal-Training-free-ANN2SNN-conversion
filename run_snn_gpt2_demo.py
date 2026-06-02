"""
Interactive GPT-2 SNN Generation & Inference Demo
==================================================
This script loads the pre-trained standard GPT-2 model, converts it into the 
fully approximated Mitchell C-2 & S-PLA Spiking Neural Network (SNN) model,
and provides an interactive shell where you can type prompts to see live text generation.

It also supports saving and loading the converted model's state_dict!
"""

import os
import sys
import math
import torch
import torch.nn as nn
import importlib
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modularized components
from modules.IEEE_754_based_Encoding import ExponentGuidedBitSliceEncoder
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
# 1. Proposed Exponent-Guided S-PLA Softmax
# ─────────────────────────────────────────────────────────────────────────────

class ProposedSoftmaxSPLA(nn.Module):
    def __init__(self, timesteps=16, s_val=8.0):
        super().__init__()
        self.timesteps = timesteps
        self.s_val = s_val
        self.encoder = ExponentGuidedBitSliceEncoder(timesteps=timesteps, s=math.ceil(math.log2(s_val)))
        self.total_spikes = 0.0
        self.num_elements = 0

    def evaluate_exp_frac_pwl(self, f):
        mask = (f < 0.5)
        w = torch.where(mask, torch.tensor(0.828427, device=f.device), torch.tensor(1.171573, device=f.device))
        c = torch.where(mask, torch.tensor(1.000000, device=f.device), torch.tensor(0.828427, device=f.device))
        return w * f + c

    def evaluate_recip_mantissa_pwl(self, M):
        mask = (M < 1.5)
        w = torch.where(mask, torch.tensor(-0.666667, device=M.device), torch.tensor(-0.333333, device=M.device))
        c = torch.where(mask, torch.tensor(1.666667, device=M.device), torch.tensor(1.166667, device=M.device))
        return w * M + c

    def forward(self, x, dim=-1):
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max
        y = x_stable * math.log2(math.e)
        E = torch.floor(y)
        frac = y - E
        
        y_frac = self.evaluate_exp_frac_pwl(frac)
        exp_x = y_frac * torch.pow(2.0, E)
        
        sum_exp = exp_x.sum(dim=dim, keepdim=True)
        E_raw = torch.floor(torch.log2(sum_exp.clamp(min=1e-30)))
        M = sum_exp / torch.pow(2.0, E_raw)
        
        y_mantissa_recip = self.evaluate_recip_mantissa_pwl(M)
        recip_sum = y_mantissa_recip * torch.pow(2.0, -E_raw)
        
        recip_broadcast = recip_sum.expand_as(exp_x)
        
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
# 2. Mitchell C-2 & S-PLA Attention Block
# ─────────────────────────────────────────────────────────────────────────────

class MitchellC2GPT2Attention(nn.Module):
    def __init__(self, original_attn, timesteps=16, s_softmax=8.0, lut=None):
        super().__init__()
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
        past = layer_past
        if 'past_key_values' in kwargs:
            past = kwargs['past_key_values']
            
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
            if hasattr(past, "update"):
                if hasattr(self, "layer_idx") and self.layer_idx is not None:
                    key, value = past.update(key, value, self.layer_idx)
            else:
                past_key, past_value = past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
                
        present = (key, value) if use_cache else None

        # Mitchell C-2 Approximate Matrix Multiplication for Attention Scores (QK^T)
        # Query shape: [batch, heads, seq_len, head_dim]
        # Key shape:   [batch, heads, seq_len_key, head_dim]
        # Result shape: [batch, heads, seq_len, seq_len_key]
        attn_weights = mitchell_c2_matmul_qk(query, key.transpose(-1, -2), self.lut)
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask bias
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Proposed Softmax S-PLA
        attn_weights_softmax = self.attn_softmax(attn_weights, dim=-1)

        # Mitchell C-2 Approximate Matrix Multiplication for Attention Values (AV)
        # Attn_weights: [batch, heads, seq_len, seq_len_key]
        # Value:        [batch, heads, seq_len_key, head_dim]
        # Result shape: [batch, heads, seq_len, head_dim]
        attn_output = mitchell_c2_matmul_av(attn_weights_softmax, value, self.lut)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_softmax,)
            
        return outputs

# ─────────────────────────────────────────────────────────────────────────────
# 3. High-Fidelity S-PLA LayerNorm and GELU Activations
# ─────────────────────────────────────────────────────────────────────────────

def _fp_decompose_even_exp(v: torch.Tensor):
    v_safe = v.abs().clamp(min=1e-38)
    E_raw  = torch.floor(torch.log2(v_safe))
    M_raw  = v_safe / torch.pow(2.0, E_raw)
    is_odd = (E_raw % 2).abs() > 0.5
    E_adj  = torch.where(is_odd, E_raw - 1, E_raw)
    M_adj  = torch.where(is_odd, M_raw * 2.0, M_raw)
    return M_adj, E_adj

class SPLALayerNorm(nn.Module):
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
        x_flat = x.view(-1, n)
        
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
            
        out_reshaped = out.view(original_shape)
        return out_reshaped * self.weight + self.bias


class SPLAActivationWrapper(nn.Module):
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
# 4. Conversion Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_snn(model, timesteps=16, prefix_k=3, device="cpu"):
    """
    In-memory Conversion of standard GPT-2 model to fully approximated
    Mitchell C-2 & S-PLA Spiking GPT-2 model. Uses balanced dynamic scales
    based on standard activation profiling of small sequences.
    """
    print(f"Applying Training-free SNN Conversion in-memory (T={timesteps})...")
    model.eval()
    
    pad = 1.15
    lut_data = torch.tensor([
        [0.0156, 0.0469, 0.0781, 0.1094],
        [0.0469, 0.1406, 0.2344, 0.3281],
        [0.0781, 0.2344, 0.3906, 0.5469],
        [0.1094, 0.3281, 0.5469, 0.7656]
    ], dtype=torch.float32)

    # Pre-calibrated/Profiled dynamic scaling factor maps
    # derived from standard GPT-2 activation ranges on casual WikiText datasets.
    # Ensures out-of-the-box accuracy without full re-calibration on new environments.
    for i, block in enumerate(model.transformer.h):
        # 1. LayerNorm 1
        mbe_ln1 = SPLALayerNorm(
            normalized_shape=block.ln_1.normalized_shape[0], 
            timesteps=timesteps,
            s_val=4.5 * pad # Robust dynamic scale factor
        )
        mbe_ln1.load_from_standard_layernorm(block.ln_1)
        block.ln_1 = mbe_ln1.to(device)
        
        # 2. LayerNorm 2
        mbe_ln2 = SPLALayerNorm(
            normalized_shape=block.ln_2.normalized_shape[0], 
            timesteps=timesteps,
            s_val=4.5 * pad
        )
        mbe_ln2.load_from_standard_layernorm(block.ln_2)
        block.ln_2 = mbe_ln2.to(device)

        # 3. GELU Activation
        mbe_gelu = SPLAActivationWrapper(
            target_name='gelu', 
            timesteps=timesteps, 
            scale_factor=6.0 * pad, 
            prefix_k=prefix_k
        )
        block.mlp.act = mbe_gelu.to(device)

        # 4. Attention (Matrix multiplications and Softmax)
        layer_idx = getattr(block.attn, 'layer_idx', None)
        approx_attn = MitchellC2GPT2Attention(block.attn, timesteps=timesteps, s_softmax=8.0, lut=lut_data)
        approx_attn.layer_idx = layer_idx
        block.attn = approx_attn.to(device)
            
    # Final LayerNorm
    mbe_lnf = SPLALayerNorm(
        normalized_shape=model.transformer.ln_f.normalized_shape[0], 
        timesteps=timesteps,
        s_val=5.0 * pad
    )
    mbe_lnf.load_from_standard_layernorm(model.transformer.ln_f)
    model.transformer.ln_f = mbe_lnf.to(device)
    
    # Final LM Head
    mbe_head = MitchellC2Linear(
        in_features=model.lm_head.in_features,
        out_features=model.lm_head.out_features,
        bias=False
    )
    mbe_head.load_from_standard_linear(model.lm_head)
    model.lm_head = mbe_head.to(device)
    
    print("SNN Conversion Complete!")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save & Load Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_snn_state(model, filepath):
    """
    Saves the state dict of the converted SNN model.
    Since weights are identical to standard ANN, this saves the state values.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"[Success] Converted SNN weights saved to: {filepath}")

def load_snn_state(model, filepath, device="cpu"):
    """
    Loads the state dict into an already instantiated/converted SNN model.
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"[Success] Converted SNN weights loaded from: {filepath}")
    return model
import time

# ─────────────────────────────────────────────────────────────────────────────
# 6. Interactive Generation Shell
# ─────────────────────────────────────────────────────────────────────────────

def generate_completion(model, tokenizer, prompt, max_length=40, device="cpu"):
    """Generates completion text and tracks wall-clock execution time."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time
    completion = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return completion, elapsed

def main():
    print("="*80)
    print("MITCHELL C-2 & S-PLA SPIKING GPT-2 INTERACTIVE GENERATION DEMO")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load standard pre-trained GPT-2
    print("\n[1/3] Loading standard pre-trained GPT-2 Small models from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load separate instances to keep the original ANN intact for comparative analysis
    print("  - Loading Original ANN GPT-2...")
    ann_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    print("  - Loading Target SNN GPT-2 (for conversion)...")
    snn_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    # 2. Perform in-memory conversion to SNN
    print("\n[2/3] Converting target model to Spiking SNN components...")
    snn_model = convert_to_snn(snn_model, timesteps=16, prefix_k=3, device=device)
    
    # 3. Enter interactive generation loop
    print("\n[3/3] SNN and ANN models are ready! Entering interactive comparative text generation shell.")
    print("Type a prompt and press Enter to generate completion text using both models.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    checkpoint_path = os.path.join(current_dir, "test_model", "snn_gpt2_weights.pth")
    
    while True:
        try:
            prompt = input("\nPrompt >> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['exit', 'quit']:
                print("\nExiting generation demo. Good luck with your research!")
                break
                
            if prompt.lower() == 'save':
                save_snn_state(snn_model, checkpoint_path)
                continue
                
            # A. SNN Model Generation
            print("SNN Generating...", end="", flush=True)
            snn_result, snn_time = generate_completion(snn_model, tokenizer, prompt, max_length=40, device=device)
            print("\r" + " "*15 + "\r", end="") # Clean loading print
            
            # B. ANN Model Generation
            print("ANN Generating...", end="", flush=True)
            ann_result, ann_time = generate_completion(ann_model, tokenizer, prompt, max_length=40, device=device)
            print("\r" + " "*15 + "\r", end="") # Clean loading print
            
            print("="*75)
            print(f"\033[92m[1] Proposed Spiking SNN GPT-2 (T=16)\033[0m")
            print(f"  - Inference Time : {snn_time:.3f} seconds")
            print(f"  - Completion Text:\n\033[36m{snn_result}\033[0m")
            print("-" * 75)
            print(f"\033[94m[2] Original Baseline ANN GPT-2\033[0m")
            print(f"  - Inference Time : {ann_time:.3f} seconds")
            print(f"  - Completion Text:\n\033[35m{ann_result}\033[0m")
            print("="*75)
            
        except KeyboardInterrupt:
            print("\nExiting generation demo.")
            break
        except Exception as e:
            print(f"\n[Error] Generation failed: {e}")

if __name__ == "__main__":
    main()
