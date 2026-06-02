"""
Mitchell C-2 Logarithmic Multiplier & Projections Module
========================================================
Implements Mitchell's Logarithmic Multiplier with 2-Term LUT-based Segmented Correction (Mitchell C-2).
Bypasses standard FP32 multipliers for Linear layers, Conv1D, and Attention MatMuls.
"""

import torch
import torch.nn as nn
import math

def decompose_float32(x):
    """
    Decomposes PyTorch float32 tensor x into:
    S: Sign bit (0 or 1)
    e: Actual exponent value (E - 127)
    M_val: Normal mantissa value in [1.0, 2.0) (or 0.0 for zero elements)
    """
    S = (x < 0.0).int()
    
    x_abs = x.abs()
    x_contiguous = x_abs.detach().contiguous()
    x_int = x_contiguous.view(torch.int32)
    
    E = (x_int >> 23) & 0xFF
    M = x_int & 0x7FFFFF
    
    e = E.clone().long() - 127
    M_val = 1.0 + M.float() / (2.0 ** 23)
    M_val = torch.where(x == 0.0, torch.zeros_like(M_val), M_val)
    
    return S, e, M_val


def mitchell_c2_multiply(A, B, lut):
    """
    Mitchell's Logarithmic Multiplier with 2-Term LUT-based Segmented Correction.
    """
    S_A, e_A, M_val_A = decompose_float32(A)
    S_B, e_B, M_val_B = decompose_float32(B)
    
    S_out = S_A ^ S_B
    
    idx_A = torch.floor((M_val_A - 1.0) * 4.0).long().clamp(0, 3)
    idx_B = torch.floor((M_val_B - 1.0) * 4.0).long().clamp(0, 3)
    
    C = lut[idx_A, idx_B]
    M_sum = M_val_A + M_val_B - 1.0 + C
    
    is_overflow = (M_sum >= 2.0)
    M_out = torch.where(is_overflow, M_sum / 2.0, M_sum)
    e_out = torch.where(is_overflow, e_A + e_B + 1, e_A + e_B)
    
    sign_factor = torch.where(S_out == 0, torch.ones_like(M_out), -torch.ones_like(M_out))
    output = sign_factor * M_out * (2.0 ** e_out)
    output = torch.where((A == 0.0) | (B == 0.0), torch.zeros_like(output), output)
    
    return output


class MitchellC2Conv1D(nn.Module):
    """
    Mitchell C-2 Logarithmic replacement for HuggingFace Transformers Conv1D.
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.Tensor(nx, nf))
        self.bias = nn.Parameter(torch.Tensor(nf))
        
        lut_data = torch.tensor([
            [0.0156, 0.0469, 0.0781, 0.1094],
            [0.0469, 0.1406, 0.2344, 0.3281],
            [0.0781, 0.2344, 0.3906, 0.5469],
            [0.1094, 0.3281, 0.5469, 0.7656]
        ], dtype=torch.float32)
        self.register_buffer('lut', lut_data)
        
    def load_from_standard_conv1d(self, conv1d):
        with torch.no_grad():
            self.weight.copy_(conv1d.weight)
            self.bias.copy_(conv1d.bias)
            
    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, self.nx) # [N, nx]
        N = x_flat.shape[0]
        
        chunk_size_n = 64
        chunk_size_f = 64
        out_flat = torch.empty(N, self.nf, device=x.device, dtype=x.dtype)
        
        for n_start in range(0, N, chunk_size_n):
            n_end = min(n_start + chunk_size_n, N)
            x_chunk = x_flat[n_start:n_end]  # [n_chunk, nx]
            n_chunk = n_end - n_start
            
            for f_start in range(0, self.nf, chunk_size_f):
                f_end = min(f_start + chunk_size_f, self.nf)
                f_chunk = f_end - f_start
                
                x_expanded = x_chunk.unsqueeze(1) # [n_chunk, 1, nx]
                w_expanded = self.weight[:, f_start:f_end].t().unsqueeze(0) # [1, f_chunk, nx]
                
                x_broadcast = x_expanded.expand(-1, f_chunk, -1)
                w_broadcast = w_expanded.expand(n_chunk, -1, -1)
                
                prod = mitchell_c2_multiply(x_broadcast, w_broadcast, self.lut)
                out_flat[n_start:n_end, f_start:f_end] = prod.sum(dim=2) + self.bias[f_start:f_end]
                
        size_out = original_shape[:-1] + (self.nf,)
        return out_flat.view(size_out)


class MitchellC2Linear(nn.Module):
    """
    Mitchell C-2 Logarithmic replacement for PyTorch nn.Linear.
    Uses nested 2D chunking to prevent OOM.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        lut_data = torch.tensor([
            [0.0156, 0.0469, 0.0781, 0.1094],
            [0.0469, 0.1406, 0.2344, 0.3281],
            [0.0781, 0.2344, 0.3906, 0.5469],
            [0.1094, 0.3281, 0.5469, 0.7656]
        ], dtype=torch.float32)
        self.register_buffer('lut', lut_data)
        
    def load_from_standard_linear(self, linear):
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)
                
    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features) # [N, in_features]
        N = x_flat.shape[0]
        
        # 2D chunking parameters to prevent OOM
        chunk_size_n = 64
        chunk_size_f = 64
        out_flat = torch.empty(N, self.out_features, device=x.device, dtype=x.dtype)
        
        for n_start in range(0, N, chunk_size_n):
            n_end = min(n_start + chunk_size_n, N)
            x_chunk = x_flat[n_start:n_end]
            n_chunk = n_end - n_start
            
            for f_start in range(0, self.out_features, chunk_size_f):
                f_end = min(f_start + chunk_size_f, self.out_features)
                f_chunk = f_end - f_start
                
                x_expanded = x_chunk.unsqueeze(1) # [n_chunk, 1, in_features]
                w_expanded = self.weight[f_start:f_end, :].unsqueeze(0) # [1, f_chunk, in_features]
                
                x_broadcast = x_expanded.expand(-1, f_chunk, -1)
                w_broadcast = w_expanded.expand(n_chunk, -1, -1)
                
                prod = mitchell_c2_multiply(x_broadcast, w_broadcast, self.lut)
                
                if self.bias is not None:
                    out_flat[n_start:n_end, f_start:f_end] = prod.sum(dim=2) + self.bias[f_start:f_end]
                else:
                    out_flat[n_start:n_end, f_start:f_end] = prod.sum(dim=2)
                    
        size_out = original_shape[:-1] + (self.out_features,)
        return out_flat.view(size_out)


def mitchell_c2_matmul_qk(q, k_t, lut):
    """
    Computes Q @ K^T matrix multiplication using Mitchell C-2.
    q:   [B, H, S, D]
    k_t: [B, H, D, S] (Key transposed)
    """
    B, H, S, D = q.shape
    key_len = k_t.shape[-1]
    out = torch.empty(B, H, S, key_len, device=q.device, dtype=q.dtype)
    chunk_size = 32
    for i in range(0, S, chunk_size):
        i_end = min(i + chunk_size, S)
        q_chunk = q[:, :, i:i_end, :].unsqueeze(3) # [B, H, chunk_size, 1, D]
        k_exp = k_t.transpose(-1, -2).unsqueeze(2) # [B, H, 1, key_len, D]
        
        prod = mitchell_c2_multiply(q_chunk, k_exp, lut) # [B, H, chunk_size, key_len, D]
        out[:, :, i:i_end, :] = prod.sum(dim=-1)
    return out


def mitchell_c2_matmul_av(attn_weights, v, lut):
    """
    Computes Attention_Weights @ V matrix multiplication using Mitchell C-2.
    attn_weights: [B, H, S, S]
    v:            [B, H, S, D]
    """
    B, H, S, key_len = attn_weights.shape
    D = v.shape[-1]
    out = torch.empty(B, H, S, D, device=attn_weights.device, dtype=attn_weights.dtype)
    chunk_size = 32
    for i in range(0, S, chunk_size):
        i_end = min(i + chunk_size, S)
        w_chunk = attn_weights[:, :, i:i_end, :].unsqueeze(4) # [B, H, chunk_size, key_len, 1]
        v_exp = v.unsqueeze(2) # [B, H, 1, key_len, D]
        
        prod = mitchell_c2_multiply(w_chunk, v_exp, lut) # [B, H, chunk_size, key_len, D]
        out[:, :, i:i_end, :] = prod.sum(dim=3)
    return out
