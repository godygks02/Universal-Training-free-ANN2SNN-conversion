"""
Spike-Driven Piecewise Linear Approximation (S-PLA) on BFE Module
==================================================================
Approximates non-linear activation functions (GELU, Sigmoid, Tanh)
using Single-Basis Ternary-Scaled (BFE) spike trains and piecewise
linear approximation (PLA) WITHOUT ANY RUNTIME MULTIPLIERS OR COMPARATORS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ---------------------------------------------------------------------------
# Surrogate gradient for STE (Straight-Through Estimator) backward pass
# ---------------------------------------------------------------------------
class SurrogateTernarySpikeSBTS(torch.autograd.Function):
    """Ternary spike with dual surrogate gradient for SB-TS.
    
    Forward: deterministic ternary quantization based on fixed thresholds.
    Backward: fast-sigmoid surrogate gradient centered at ±V_th.
    """
    @staticmethod
    def forward(ctx, u, v_th):
        ctx.save_for_backward(u, v_th)
        s = torch.zeros_like(u)
        s[u >= v_th] = 1.0
        s[u <= -v_th] = -1.0
        return s

    @staticmethod
    def backward(ctx, grad_output):
        u, v_th = ctx.saved_tensors
        gamma = 2.0
        sg_pos = gamma / (1.0 + gamma * torch.abs(u - v_th)) ** 2
        sg_neg = gamma / (1.0 + gamma * torch.abs(u + v_th)) ** 2
        grad_u = grad_output * (sg_pos + sg_neg)
        return grad_u, None  # v_th is not learnable


# ---------------------------------------------------------------------------
# Core SB-TS Neuron Module
# ---------------------------------------------------------------------------
class SBTSNeuron(nn.Module):
    """
    Single-Basis Ternary-Scaled (SB-TS) Neuron.
    
    A 100% deterministic, training-free encoder that converts normalized
    real values in [-1, 1] into a sequence of ternary spikes {-1, 0, +1}.
    """
    def __init__(self, timesteps=4):
        super().__init__()
        self.timesteps = timesteps
        
        t_indices = torch.arange(1, timesteps + 1, dtype=torch.float32)
        self.register_buffer('d', 2.0 ** (-t_indices))         # intensity & reset
        self.register_buffer('v_th', 2.0 ** (-(t_indices + 1)))  # threshold
        
        self.ternary_spike = SurrogateTernarySpikeSBTS.apply

    def forward(self, x_norm, return_sequences=False):
        u = x_norm.clone()
        spike_sequence = [] if return_sequences else None
        decoded = torch.zeros_like(x_norm)

        for t in range(self.timesteps):
            d_t = self.d[t]
            v_th_t = self.v_th[t]

            s_t = self.ternary_spike(u, v_th_t)
            decoded = decoded + s_t * d_t
            u = u - s_t * d_t

            if return_sequences:
                spike_sequence.append(s_t)

        if return_sequences:
            return decoded, torch.stack(spike_sequence, dim=0)
        return decoded

    def get_max_error(self):
        return 2.0 ** (-(self.timesteps + 1))


# ---------------------------------------------------------------------------
# Offline Segment Calibration
# ---------------------------------------------------------------------------
def calibrate_pla_segments(target_func, scale_factor, timesteps, prefix_k, num_grid_points=100000, device='cpu'):
    """
    Offline calibration of PLA segments based on BFE prefix routing.
    """
    neuron = SBTSNeuron(timesteps=timesteps).to(device)
    
    x_grid = np.linspace(-1.0, 1.0, num_grid_points)
    x_grid_torch = torch.from_numpy(x_grid).float().to(device).unsqueeze(1)
    
    with torch.no_grad():
        _, spikes = neuron(x_grid_torch, return_sequences=True)
        
    d_prefix = neuron.d[:prefix_k]
    spikes_prefix = spikes[:prefix_k, :, 0]
    x_rec_prefix = torch.sum(spikes_prefix * d_prefix.unsqueeze(1), dim=0)
    
    x_rec_prefix_np = x_rec_prefix.cpu().numpy()
    x_rec_prefix_rounded = np.round(x_rec_prefix_np, decimals=8)
    
    unique_vals = np.unique(x_rec_prefix_rounded)
    M = len(unique_vals)
    
    slopes = np.zeros(M)
    intercepts = np.zeros(M)
    
    for idx, val in enumerate(unique_vals):
        mask = (x_rec_prefix_rounded == val)
        x_seg = x_grid[mask] * scale_factor
        y_seg = target_func(x_seg)
        
        if len(x_seg) >= 2:
            a, b = np.polyfit(x_seg, y_seg, 1)
        elif len(x_seg) == 1:
            a = 0.0
            b = y_seg[0]
        else:
            a = 0.0
            b = 0.0
            
        slopes[idx] = a
        intercepts[idx] = b
        
    boundaries = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    return unique_vals, slopes, intercepts, boundaries


# ---------------------------------------------------------------------------
# Core Spike-Driven PLA Activation Module
# ---------------------------------------------------------------------------
class SBTSPLAActivation(nn.Module):
    """
    Piecewise Linear Approximation (PLA) using BFE spike-based prefix routing.
    f(x) ≈ a_i * x + b_i
    Where segment index 'i' is determined solely by the first K spikes of BFE.
    """
    def __init__(self, target_name='gelu', timesteps=16, scale_factor=3.0, prefix_k=3, num_grid_points=100000):
        super().__init__()
        self.target_name = target_name.lower()
        self.timesteps = timesteps
        self.scale_factor = scale_factor
        self.prefix_k = prefix_k
        
        targets = {
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
            'tanh': np.tanh,
            'gelu': lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        }
        if self.target_name not in targets:
            raise ValueError(f"Unknown target function: {target_name}")
        self.target_func = targets[self.target_name]
        
        self.encoder = SBTSNeuron(timesteps=timesteps)
        
        unique_vals, slopes, intercepts, boundaries = calibrate_pla_segments(
            self.target_func, scale_factor, timesteps, prefix_k, num_grid_points
        )
        
        self.register_buffer('unique_vals', torch.tensor(unique_vals, dtype=torch.float32))
        self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))
        self.register_buffer('intercepts', torch.tensor(intercepts, dtype=torch.float32))
        self.register_buffer('boundaries', torch.tensor(boundaries, dtype=torch.float32))
        
    def forward(self, x, return_details=False):
        x_norm = (x / self.scale_factor).clamp(-1.0, 1.0)
        x_rec_norm, spikes = self.encoder(x_norm, return_sequences=True)
        
        d_prefix = self.encoder.d[:self.prefix_k].view(self.prefix_k, *([1] * x.dim()))
        spikes_prefix = spikes[:self.prefix_k]
        x_rec_prefix = torch.sum(spikes_prefix * d_prefix, dim=0)
        
        seg_idx = torch.bucketize(x_rec_prefix, self.boundaries)
        
        a_i = self.slopes[seg_idx]
        b_i = self.intercepts[seg_idx]
        
        A_i = a_i * self.scale_factor
        d_broadcast = self.encoder.d.view(-1, *([1] * x.dim()))
        
        term_all = spikes * (A_i.unsqueeze(0) * d_broadcast)
        out_step = b_i + torch.sum(term_all, dim=0)
        
        if return_details:
            return out_step, spikes, seg_idx, x_rec_prefix
        return out_step
