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
import importlib

# Robust imports for sibling modules
try:
    from modules.IEEE_754_based_Encoding import IEEE754_based_encoder
except ImportError:
    from IEEE_754_based_Encoding import IEEE754_based_encoder

try:
    mitchell_c2_approx = importlib.import_module("modules.mitchell_c-2_approx")
except ImportError:
    mitchell_c2_approx = importlib.import_module("mitchell_c-2_approx")

mitchell_c2_multiply = mitchell_c2_approx.mitchell_c2_multiply


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
# Core SB-TS Neuron Module (Legacy reference / BFE)
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
def calibrate_pla_segments(target_func, scale_factor, timesteps, prefix_k, min_e_routing=-5, num_grid_points=100000, device='cpu'):
    """
    Offline calibration of PLA segments based on IEEE 754 prefix routing.
    """
    encoder = IEEE754_based_encoder(timesteps=timesteps, s=0).to(device)
    
    x_grid = np.linspace(-1.0, 1.0, num_grid_points)
    x_grid_torch = torch.from_numpy(x_grid).float().to(device).unsqueeze(1)
    
    with torch.no_grad():
        S, spikes, e = encoder(x_grid_torch)
        
    d_prefix = encoder.d[:prefix_k].view(prefix_k, 1, 1)
    spikes_prefix = spikes[:prefix_k]
    M_rec_prefix = torch.sum(spikes_prefix * d_prefix, dim=0)
    
    # Exponent Clamping for prefix routing to prevent segment explosion / OOM
    e_prefix = torch.clamp(e, min=min_e_routing)
    scale_factor_e = torch.pow(2.0, e_prefix.float())
    sign_factor = torch.where(S == 0, torch.ones_like(M_rec_prefix), -torch.ones_like(M_rec_prefix))
    
    x_rec_prefix = M_rec_prefix * scale_factor_e * sign_factor
    # Route all values below min_e_routing to the same central segment (0.0)
    x_rec_prefix = torch.where(e >= min_e_routing, x_rec_prefix, torch.zeros_like(x_rec_prefix))
    
    x_rec_prefix_np = x_rec_prefix[:, 0].cpu().numpy()
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
class IEEE754_based_SPLA(nn.Module):
    """
    Piecewise Linear Approximation (PLA) using IEEE 754 spike-based prefix routing.
    f(x) ≈ a_i * x + b_i
    Where segment index 'i' is determined solely by the first K spikes of IEEE 754 encoding.
    """
    def __init__(self, target_name='gelu', timesteps=16, scale_factor=3.0, prefix_k=3, min_e_routing=-5, num_grid_points=100000):
        super().__init__()
        self.target_name = target_name.lower()
        self.timesteps = timesteps
        self.scale_factor = scale_factor
        self.prefix_k = prefix_k
        self.min_e_routing = min_e_routing
        
        targets = {
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
            'tanh': np.tanh,
            'gelu': lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        }
        if self.target_name not in targets:
            raise ValueError(f"Unknown target function: {target_name}")
        self.target_func = targets[self.target_name]
        
        # Use IEEE 754 based encoder
        self.encoder = IEEE754_based_encoder(timesteps=timesteps, s=0)
        
        unique_vals, slopes, intercepts, boundaries = calibrate_pla_segments(
            self.target_func, scale_factor, timesteps, prefix_k, min_e_routing, num_grid_points
        )
        
        self.register_buffer('unique_vals', torch.tensor(unique_vals, dtype=torch.float32))
        self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))
        self.register_buffer('intercepts', torch.tensor(intercepts, dtype=torch.float32))
        self.register_buffer('boundaries', torch.tensor(boundaries, dtype=torch.float32))
        
    def forward(self, x, return_details=False):
        x_norm = (x / self.scale_factor).clamp(-1.0, 1.0)
        S, spikes, e = self.encoder(x_norm)
        
        d_prefix = self.encoder.d[:self.prefix_k].view(self.prefix_k, *([1] * x.dim()))
        spikes_prefix = spikes[:self.prefix_k]
        M_rec_prefix = torch.sum(spikes_prefix * d_prefix, dim=0)
        
        # Exponent Clamping for prefix routing to prevent segment explosion / OOM
        e_prefix = torch.clamp(e, min=self.min_e_routing)
        scale_factor_e = torch.pow(2.0, e_prefix.float())
        sign_factor = torch.where(S == 0, torch.ones_like(M_rec_prefix), -torch.ones_like(M_rec_prefix))
        
        x_rec_prefix = M_rec_prefix * scale_factor_e * sign_factor
        # Route all values below min_e_routing to the same central segment (0.0)
        x_rec_prefix = torch.where(e >= self.min_e_routing, x_rec_prefix, torch.zeros_like(x_rec_prefix))
        
        seg_idx = torch.bucketize(x_rec_prefix, self.boundaries)
        
        a_i = self.slopes[seg_idx]
        b_i = self.intercepts[seg_idx]
        
        # Apply Exponent Wired-Alignment: scale slope by 2^e (actual un-clamped exponent)
        scale_factor_actual_e = torch.pow(2.0, e.float())
        A_i = a_i * self.scale_factor * scale_factor_actual_e
        d_broadcast = self.encoder.d.view(-1, *([1] * x.dim()))
        
        sign_broadcast = sign_factor.unsqueeze(0)
        term_all = spikes * (A_i.unsqueeze(0) * d_broadcast) * sign_broadcast
        out_step = b_i + torch.sum(term_all, dim=0)
        
        if return_details:
            return out_step, spikes, seg_idx, x_rec_prefix
        return out_step


# ---------------------------------------------------------------------------
# Helper function for even exponent decomposition
# ---------------------------------------------------------------------------
def _fp_decompose_even_exp(v: torch.Tensor):
    """Decomposes positive v into M_adj * 2^E_adj where E_adj is always even."""
    v_safe = v.abs().clamp(min=1e-38)
    E_raw  = torch.floor(torch.log2(v_safe))
    M_raw  = v_safe / torch.pow(2.0, E_raw)
    is_odd = (E_raw % 2).abs() > 0.5
    E_adj  = torch.where(is_odd, E_raw - 1, E_raw)
    M_adj  = torch.where(is_odd, M_raw * 2.0, M_raw)
    return M_adj, E_adj


# ---------------------------------------------------------------------------
# Modularized S-PLA LayerNorm
# ---------------------------------------------------------------------------
class SPLALayerNorm(nn.Module):
    """
    LayerNorm approximating centering, squaring, and inv_sqrt using S-PLA segments,
    and final scaling utilizing IEEE 754 Exponent-Guided Bit-Slice spiking interaction.
    """
    def __init__(self, normalized_shape, eps=1e-5, s_val=3.0, timesteps=16, approx_square='pwl'):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.s_val = s_val
        self.timesteps = timesteps
        self.approx_square = approx_square
        
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        
        # Exponent guided bit slice encoder (renamed)
        self.proposed_encoder = IEEE754_based_encoder(timesteps=timesteps, s=math.ceil(math.log2(s_val)))
        
        # Mitchell C-2 correction LUT (4x4 symmetric correction matrix)
        lut_data = torch.tensor([
            [0.0156, 0.0469, 0.0781, 0.1094],
            [0.0469, 0.1406, 0.2344, 0.3281],
            [0.0781, 0.2344, 0.3906, 0.5469],
            [0.1094, 0.3281, 0.5469, 0.7656]
        ], dtype=torch.float32)
        self.register_buffer('lut', lut_data)
        
        # Energy metrics tracking (1D S-PLA Firing Rates)
        self.total_sq_spikes = 0.0
        self.total_v_spikes = 0.0
        self.num_elements = 0
        self.num_samples = 0
        
    def load_from_standard_layernorm(self, ln):
        with torch.no_grad():
            self.weight.copy_(ln.weight)
            self.bias.copy_(ln.bias)
            
    def evaluate_square_pwl(self, a):
        # 8-segment PWL Square Approximation
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
        # 8-segment PWL InvSqrt Mantissa Approximation
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
        
        # 1. Mean & Centered Diff
        mu = x_flat.mean(dim=dim, keepdim=True)
        centered = x_flat - mu
        
        if self.approx_square == 'pwl':
            s = centered.abs().amax(dim=dim, keepdim=True).detach().clamp(min=1e-6)
            a = (centered / s).clamp(-1, 1)
            a_sq = self.evaluate_square_pwl(a)
            centered_sq = a_sq * (s ** 2)
        else: # 'mitchell'
            centered_sq = mitchell_c2_multiply(centered, centered, self.lut)
            
        sum_sq = centered_sq.sum(dim=dim, keepdim=True)
        
        # 2. Inverse Square Root
        v = sum_sq + n * self.eps
        M_adj, E_adj = _fp_decompose_even_exp(v)
        
        inv_sqrt_M = self.evaluate_invsqrt_mantissa_pwl(M_adj)
        shift = torch.pow(2.0, -E_adj / 2.0)
        inv_sqrt = inv_sqrt_M * shift
        inv_sqrt_var = inv_sqrt * math.sqrt(n)
        
        # 3. Dynamic Multiplication using Mitchell C-2 Multiplier
        out = mitchell_c2_multiply(centered, inv_sqrt_var.expand_as(centered), self.lut)
        
        # 4. Track Firing Rates for Energy Metrics (1D S-PLA)
        with torch.no_grad():
            if self.approx_square == 'pwl':
                _, spikes_sq, _ = self.proposed_encoder(a)
                self.total_sq_spikes += spikes_sq.abs().float().sum().item()
            else:
                self.total_sq_spikes += 0.0
                
            _, spikes_v, _ = self.proposed_encoder(v)
            self.total_v_spikes += spikes_v.abs().float().sum().item()
            self.num_elements += x_flat.numel()
            self.num_samples += x_flat.shape[0]
            
        out_reshaped = out.view(original_shape)
        return out_reshaped * self.weight + self.bias


# ---------------------------------------------------------------------------
# Modularized S-PLA Softmax
# ---------------------------------------------------------------------------
class ProposedSoftmaxSPLA(nn.Module):
    """
    Proposed Softmax using IEEE 754 Exponent-Guided Bit-Slice S-PLA.
    Includes active spike tracking for energy compilation.
    """
    def __init__(self, timesteps=16, s_val=8.0):
        super().__init__()
        self.timesteps = timesteps
        self.s_val = s_val
        self.encoder = IEEE754_based_encoder(timesteps=timesteps, s=math.ceil(math.log2(s_val)))
        
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
        _, spikes_exp, _ = self.encoder(exp_x)
        _, spikes_recip, _ = self.encoder(recip_broadcast)
        
        with torch.no_grad():
            self.total_spikes += spikes_exp.abs().float().sum().item() + spikes_recip.abs().float().sum().item()
            self.num_elements += exp_x.numel() + recip_broadcast.numel()
            
        out = exp_x * recip_broadcast
        out_sum = out.sum(dim=dim, keepdim=True).clamp(min=1e-8)
        out = out / out_sum
        
        return out


# ---------------------------------------------------------------------------
# Modularized S-PLA Activation Wrapper
# ---------------------------------------------------------------------------
class SPLAActivationWrapper(nn.Module):
    """
    SPLA Activation Wrapper driven by IEEE 754 encoder spikes.
    Tracks firing rate and spike totals to compute exact INT dynamic additions.
    """
    def __init__(self, target_name='gelu', timesteps=16, scale_factor=3.0, prefix_k=3, min_e_routing=-5):
        super().__init__()
        self.spla = IEEE754_based_SPLA(target_name=target_name, timesteps=timesteps, scale_factor=scale_factor, prefix_k=prefix_k, min_e_routing=min_e_routing)
        self.total_spikes = 0.0
        self.num_elements = 0
        
    def forward(self, x):
        out_step, spikes, _, _ = self.spla(x, return_details=True)
        
        with torch.no_grad():
            # Use absolute sum of binary spikes
            self.total_spikes += spikes.abs().float().sum().item()
            self.num_elements += x.numel()
            
        return out_step


# Legacy alias for backward compatibility
SBTSPLAActivation = IEEE754_based_SPLA
