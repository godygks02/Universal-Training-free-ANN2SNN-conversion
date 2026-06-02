"""
IEEE 754 Exponent-Guided Bit-Slice Spiking Encoder Module
=========================================================
Implements the proposed hardware-wired Zero-Cost spiking encoder.
Extracts binary fixed-point spike trains directly from FP32 tensors 
using hardware-wired bitwise shift and mask operations.
"""

import torch
import torch.nn as nn

class ExponentGuidedBitSliceEncoder(nn.Module):
    """
    IEEE 754 Exponent-Guided Bit-Slice Spiking Encoder.
    
    Extracts the binary fixed-point representation of a Float32 tensor directly
    using bitwise shift and mask operations, emulating a zero-cost hardware-wired shift bus.
    
    Assumes a dynamic range covered by 's' integer bits.
    Max representable value = 2^s - 2^(-T+s)
    """
    def __init__(self, timesteps=16, s=2):
        super().__init__()
        self.timesteps = timesteps
        self.s = s  # Scale factor power (2^s)
        
        # Register the temporal weights as a buffer: d[t] = 2^(-t + s)
        t_indices = torch.arange(1, timesteps + 1, dtype=torch.float32)
        self.register_buffer('d', 2.0 ** (-t_indices + s))

    def forward(self, x):
        """
        Encodes the float32 tensor x into a sign bit S and binary spike train s_x.
        
        Args:
            x: PyTorch Float32 Tensor of any shape.
        Returns:
            S: Sign bits (0 for positive, 1 for negative). Shape: [*x.shape]
            spikes: Binary spike trains s_x[t] in {0, 1}. Shape: [T, *x.shape]
        """
        device = x.device
        
        # Ensure contiguous, detached, and cast to int32 view
        x_contiguous = x.detach().contiguous()
        x_int = x_contiguous.view(torch.int32)
        
        S = (x_int >> 31) & 1  # Sign bit (1-bit)
        E = (x_int >> 23) & 0xFF  # Exponent field (8-bit)
        M = (x_int & 0x7FFFFF).to(torch.int32)  # Force integer type to prevent CPU shift errors
        
        # Compute actual exponent: e = E.clone().long() - 127
        e = E.clone().long() - 127
        
        # 1. Create a vectorized time index tensor `t` of shape [T, 1, 1...] matching x's dimensions
        t = torch.arange(1, self.timesteps + 1, device=device).view(-1, *([1] * x.dim()))
        
        # 2. Vectorize shift calculations across all timesteps T: shift shape is [T, *x.shape]
        # e and M must be unsqueezed at dim 0 to broadcast with t
        shift = 23 - t + self.s - e.unsqueeze(0)
        
        # Mask shifts that fall outside the 23-bit mantissa range to prevent errors
        valid_mask = (shift >= 0) & (shift < 23)
        safe_shift = torch.where(valid_mask, shift, torch.zeros_like(shift)).to(torch.int32)
        
        # Slice mantissa bits in parallel
        mantissa_bit = (M.unsqueeze(0) >> safe_shift) & 1
        mantissa_bit = torch.where(valid_mask, mantissa_bit, torch.zeros_like(implicit_bit if 'implicit_bit' in locals() else mantissa_bit)) # Safe fallback
        mantissa_bit = torch.where(valid_mask, mantissa_bit, torch.zeros_like(mantissa_bit))
        
        # Implicit leading bit (occurs exactly when the exponent matches the target power: e == s - t)
        implicit_bit = (e.unsqueeze(0) == (self.s - t)).int()
        
        # If e == s - t, the bit is the implicit leading 1. Otherwise, it is the extracted mantissa bit.
        spikes = torch.where(e.unsqueeze(0) == (self.s - t), implicit_bit, mantissa_bit)
        
        # Zero out spikes for absolute values smaller than the minimum precision or true zero
        spikes = torch.where(x.unsqueeze(0).abs() < 1e-15, torch.zeros_like(spikes), spikes)
        
        return S, spikes.float()

    def decode(self, S, spikes):
        """
        Decodes the sign bit S and binary spike train back into real values.
        Matches the hardware Sign-Controlled ADD/SUBTRACT synapse logic:
            x_approx = (-1)^S * sum(s_x[t] * d[t])
        """
        # Temporal summation: sum_t (s_x[t] * d[t])
        d_broadcast = self.d.view(-1, *([1] * S.dim()))
        unsigned_sum = torch.sum(spikes * d_broadcast, dim=0)
        
        # Apply sign control: ADD if S == 0, SUBTRACT if S == 1
        sign_factor = torch.where(S == 0, torch.ones_like(unsigned_sum), -torch.ones_like(unsigned_sum))
        return unsigned_sum * sign_factor
