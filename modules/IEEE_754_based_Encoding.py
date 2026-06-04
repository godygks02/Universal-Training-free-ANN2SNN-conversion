"""
IEEE 754 Exponent-Guided Bit-Slice Spiking Encoder Module
=========================================================
Implements the proposed hardware-wired Zero-Cost spiking encoder.
Extracts binary fixed-point spike trains directly from FP32 tensors 
using hardware-wired bitwise shift and mask operations.
"""

import torch
import torch.nn as nn

class IEEE754_based_encoder(nn.Module):
    """
    IEEE 754 Exponent-Guided Bit-Slice Spiking Encoder.
    
    Extracts the binary fixed-point representation of a Float32 tensor directly
    using bitwise shift and mask operations, emulating a zero-cost hardware-wired shift bus.
    
    Assumes a dynamic range covered by 's' integer bits.
    Max representable value = 2^s - 2^(-T+s)
    """
    def __init__(self, timesteps=16, s=0):
        super().__init__()
        self.timesteps = timesteps
        self.s = s  # Kept for backward compatibility, not used for shifting anymore
        
        # Register the temporal weights of pure mantissa: d[t] = 2^(-t + 1)
        t_indices = torch.arange(1, timesteps + 1, dtype=torch.float32)
        self.register_buffer('d', 2.0 ** (-t_indices + 1))

    def forward(self, x):
        """
        Encodes the float32 tensor x into a sign bit S, binary spike train s_x, and exponent e.
        
        Args:
            x: PyTorch Float32 Tensor of any shape.
        Returns:
            S: Sign bits (0 for positive, 1 for negative). Shape: [*x.shape]
            spikes: Binary spike trains s_x[t] in {0, 1}. Shape: [T, *x.shape]
            e: Exponent tensor. Shape: [*x.shape]
        """
        device = x.device
        
        # Ensure contiguous, detached, and cast to int32 view
        x_contiguous = x.detach().contiguous()
        x_int = x_contiguous.view(torch.int32)
        
        S = (x_int >> 31) & 1  # Sign bit (1-bit)
        E = (x_int >> 23) & 0xFF  # Exponent field (8-bit)
        M = (x_int & 0x7FFFFF).to(torch.int32)  # Force integer type
        
        # Compute actual exponent: e = E - 127
        e = E.clone().long() - 127
        
        # 1. Create a vectorized time index tensor `t` of shape [T, 1, 1...] matching x's dimensions
        t = torch.arange(1, self.timesteps + 1, device=device).view(-1, *([1] * x.dim()))
        
        # 2. At t=1, spike is the implicit leading bit (1 for normal numbers, 0 for zero/subnormal)
        is_normal = (E > 0).int()
        
        # 3. At t >= 2, we extract mantissa bits. Shift is 24 - t.
        shift = 24 - t
        safe_shift = torch.clamp(shift, 0, 22)
        mantissa_bit = (M.unsqueeze(0) >> safe_shift) & 1
        
        # Combine implicit bit and mantissa bits
        spikes = torch.where(t == 1, is_normal.unsqueeze(0), mantissa_bit)
        
        # Zero out spikes for absolute values smaller than the minimum precision or true zero
        spikes = torch.where(x.unsqueeze(0).abs() < 1e-15, torch.zeros_like(spikes), spikes)
        
        return S, spikes.float(), e

    def decode(self, S, spikes, e):
        """
        Decodes the sign bit S, binary spike train, and exponent e back into real values.
        x_approx = (-1)^S * M_rec * 2^e
        """
        # Temporal summation for mantissa: sum_t (s_x[t] * d[t])
        d_broadcast = self.d.view(-1, *([1] * S.dim()))
        M_rec = torch.sum(spikes * d_broadcast, dim=0)
        
        # Apply exponent: scale by 2^e using safe bitwise float view hack (zero latency on GPU/CPU)
        scale_factor = ((e + 127).to(torch.int32) << 23).view(torch.float32)
        
        # Apply sign control: ADD if S == 0, SUBTRACT if S == 1
        sign_factor = torch.where(S == 0, torch.ones_like(M_rec), -torch.ones_like(M_rec))
        return M_rec * scale_factor * sign_factor

# Alias for backward compatibility
ExponentGuidedBitSliceEncoder = IEEE754_based_encoder
