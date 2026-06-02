"""
Fashion-MNIST VAE Spiking Image Reconstruction & Generation using Mitchell C-2 & S-PLA
=============================================================================
This script introduces the Training-Free ANN-to-SNN conversion methods to the
GENERATIVE domain by implementing a deeper, higher-capacity VAE on Fashion-MNIST.

It:
1. Loads the pre-trained Fashion-MNIST VAE (ANN).
2. Converts the VAE to a Spiking VAE (SNN) in-memory using Mitchell C-2 & S-PLA.
3. Evaluates reconstruction quality (MSE) and compares ANN vs. SNN.
4. Generates brand new images by sampling latent noise z ~ N(0, I) through the SNN Decoder!
5. Visualizes reconstructions and new generations in beautiful comparative grids.
6. Conducts a complete theoretical dynamic energy savings analysis.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_utils import get_device, ToyVAE

# Import modularized components
from modules.IEEE_754_based_Encoding import ExponentGuidedBitSliceEncoder
mitchell_c2_approx = importlib.import_module("modules.mitchell_c-2_approx")
spla_module = importlib.import_module("modules.S-PLA")

decompose_float32 = mitchell_c2_approx.decompose_float32
mitchell_c2_multiply = mitchell_c2_approx.mitchell_c2_multiply
MitchellC2Linear = mitchell_c2_approx.MitchellC2Linear
SBTSPLAActivation = spla_module.SBTSPLAActivation


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spiking LayerNorm & GELU Activation Wrappers for Generative Networks
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
    """
    Spiking Piecewise Linear LayerNorm tailored for Generative Decoder/Encoder feedforward pathways.
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
        
        mu = x.mean(dim=dim, keepdim=True)
        centered = x - mu
        
        s = centered.abs().amax(dim=dim, keepdim=True).detach().clamp(min=1e-6)
        a = (centered / s).clamp(-1, 1)
        
        a_sq = self.evaluate_square_pwl(a)
        centered_sq = a_sq * (s ** 2)
        sum_sq = centered_sq.sum(dim=dim, keepdim=True)
        
        v = sum_sq + n * self.eps
        M_adj, E_adj = _fp_decompose_even_exp(v)
        
        inv_sqrt_M = self.evaluate_invsqrt_mantissa_pwl(M_adj)
        shift = torch.pow(2.0, -E_adj / 2.0)
        inv_sqrt = inv_sqrt_M * shift
        inv_sqrt_var = inv_sqrt * math.sqrt(n)
        
        out = mitchell_c2_multiply(centered, inv_sqrt_var.expand_as(centered), self.lut)
        
        with torch.no_grad():
            _, spikes_sq = self.proposed_encoder(a)
            _, spikes_v = self.proposed_encoder(v)
            self.total_sq_spikes += spikes_sq.abs().float().sum().item()
            self.total_v_spikes += spikes_v.abs().float().sum().item()
            self.num_elements += x.numel()
            self.num_samples += x.shape[0]
            
        return out * self.weight + self.bias


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


def generate_fashion_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Fashion-MNIST Data Loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples.")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hybrid SNN Conversion & Module Replacements for VAE
# ─────────────────────────────────────────────────────────────────────────────

class SpikingVAE(nn.Module):
    """
    Converted Spiking VAE using Mitchell C-2 & S-PLA Approximations (Deeper Architecture).
    """
    def __init__(self, ann_model, timesteps=16, s_ln1=4.0, s_ln2=4.0, s_ln3=4.0, s_ln4=4.0,
                 s_gelu1=4.0, s_gelu2=4.0, s_gelu3=4.0, s_gelu4=4.0, prefix_k=3,
                 approx_linear=True, approx_ln=True, approx_gelu=True):
        super().__init__()
        self.latent_dim = ann_model.latent_dim
        self.input_dim = ann_model.input_dim
        self.hidden_dim1 = ann_model.hidden_dim1
        self.hidden_dim2 = ann_model.hidden_dim2
        
        # 1. Encoder Pathway
        if approx_linear:
            self.fc1 = MitchellC2Linear(ann_model.fc1.in_features, ann_model.fc1.out_features)
            self.fc1.load_from_standard_linear(ann_model.fc1)
        else:
            self.fc1 = nn.Linear(ann_model.fc1.in_features, ann_model.fc1.out_features)
            with torch.no_grad():
                self.fc1.weight.copy_(ann_model.fc1.weight)
                self.fc1.bias.copy_(ann_model.fc1.bias)
                
        if approx_ln:
            self.ln1 = SPLALayerNorm(ann_model.ln1.normalized_shape[0], timesteps=timesteps, s_val=s_ln1)
            self.ln1.load_from_standard_layernorm(ann_model.ln1)
        else:
            self.ln1 = nn.LayerNorm(ann_model.ln1.normalized_shape[0])
            with torch.no_grad():
                self.ln1.weight.copy_(ann_model.ln1.weight)
                self.ln1.bias.copy_(ann_model.ln1.bias)
                
        if approx_gelu:
            self.gelu1 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu1, prefix_k=prefix_k)
        else:
            self.gelu1 = nn.GELU()
            
        if approx_linear:
            self.fc2 = MitchellC2Linear(ann_model.fc2.in_features, ann_model.fc2.out_features)
            self.fc2.load_from_standard_linear(ann_model.fc2)
        else:
            self.fc2 = nn.Linear(ann_model.fc2.in_features, ann_model.fc2.out_features)
            with torch.no_grad():
                self.fc2.weight.copy_(ann_model.fc2.weight)
                self.fc2.bias.copy_(ann_model.fc2.bias)
                
        if approx_ln:
            self.ln2 = SPLALayerNorm(ann_model.ln2.normalized_shape[0], timesteps=timesteps, s_val=s_ln2)
            self.ln2.load_from_standard_layernorm(ann_model.ln2)
        else:
            self.ln2 = nn.LayerNorm(ann_model.ln2.normalized_shape[0])
            with torch.no_grad():
                self.ln2.weight.copy_(ann_model.ln2.weight)
                self.ln2.bias.copy_(ann_model.ln2.bias)
                
        if approx_gelu:
            self.gelu2 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu2, prefix_k=prefix_k)
        else:
            self.gelu2 = nn.GELU()

        # Encoder FC Mu & FC LogVar
        if approx_linear:
            self.fc_mu = MitchellC2Linear(ann_model.fc_mu.in_features, ann_model.fc_mu.out_features)
            self.fc_mu.load_from_standard_linear(ann_model.fc_mu)
            
            self.fc_logvar = MitchellC2Linear(ann_model.fc_logvar.in_features, ann_model.fc_logvar.out_features)
            self.fc_logvar.load_from_standard_linear(ann_model.fc_logvar)
        else:
            self.fc_mu = nn.Linear(ann_model.fc_mu.in_features, ann_model.fc_mu.out_features)
            self.fc_logvar = nn.Linear(ann_model.fc_logvar.in_features, ann_model.fc_logvar.out_features)
            with torch.no_grad():
                self.fc_mu.weight.copy_(ann_model.fc_mu.weight)
                self.fc_mu.bias.copy_(ann_model.fc_mu.bias)
                self.fc_logvar.weight.copy_(ann_model.fc_logvar.weight)
                self.fc_logvar.bias.copy_(ann_model.fc_logvar.bias)
                
        # 2. Decoder Pathway
        if approx_linear:
            self.fc3 = MitchellC2Linear(ann_model.fc3.in_features, ann_model.fc3.out_features)
            self.fc3.load_from_standard_linear(ann_model.fc3)
        else:
            self.fc3 = nn.Linear(ann_model.fc3.in_features, ann_model.fc3.out_features)
            with torch.no_grad():
                self.fc3.weight.copy_(ann_model.fc3.weight)
                self.fc3.bias.copy_(ann_model.fc3.bias)
                
        if approx_ln:
            self.ln3 = SPLALayerNorm(ann_model.ln3.normalized_shape[0], timesteps=timesteps, s_val=s_ln3)
            self.ln3.load_from_standard_layernorm(ann_model.ln3)
        else:
            self.ln3 = nn.LayerNorm(ann_model.ln3.normalized_shape[0])
            with torch.no_grad():
                self.ln3.weight.copy_(ann_model.ln3.weight)
                self.ln3.bias.copy_(ann_model.ln3.bias)
                
        if approx_gelu:
            self.gelu3 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu3, prefix_k=prefix_k)
        else:
            self.gelu3 = nn.GELU()
            
        if approx_linear:
            self.fc4 = MitchellC2Linear(ann_model.fc4.in_features, ann_model.fc4.out_features)
            self.fc4.load_from_standard_linear(ann_model.fc4)
        else:
            self.fc4 = nn.Linear(ann_model.fc4.in_features, ann_model.fc4.out_features)
            with torch.no_grad():
                self.fc4.weight.copy_(ann_model.fc4.weight)
                self.fc4.bias.copy_(ann_model.fc4.bias)
                
        if approx_ln:
            self.ln4 = SPLALayerNorm(ann_model.ln4.normalized_shape[0], timesteps=timesteps, s_val=s_ln4)
            self.ln4.load_from_standard_layernorm(ann_model.ln4)
        else:
            self.ln4 = nn.LayerNorm(ann_model.ln4.normalized_shape[0])
            with torch.no_grad():
                self.ln4.weight.copy_(ann_model.ln4.weight)
                self.ln4.bias.copy_(ann_model.ln4.bias)
                
        if approx_gelu:
            self.gelu4 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu4, prefix_k=prefix_k)
        else:
            self.gelu4 = nn.GELU()
            
        if approx_linear:
            self.fc5 = MitchellC2Linear(ann_model.fc5.in_features, ann_model.fc5.out_features)
            self.fc5.load_from_standard_linear(ann_model.fc5)
        else:
            self.fc5 = nn.Linear(ann_model.fc5.in_features, ann_model.fc5.out_features)
            with torch.no_grad():
                self.fc5.weight.copy_(ann_model.fc5.weight)
                self.fc5.bias.copy_(ann_model.fc5.bias)

    def encode(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.gelu1(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.gelu2(h)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.fc3(z)
        h = self.ln3(h)
        h = self.gelu3(h)
        h = self.fc4(h)
        h = self.ln4(h)
        h = self.gelu4(h)
        return torch.sigmoid(self.fc5(h))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# 3. Activation Profiler
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_vae_ranges(model, loader, device, num_batches=5):
    model.eval()
    ranges = {'ln1_in': [], 'gelu1_in': [], 'ln2_in': [], 'gelu2_in': [],
              'ln3_in': [], 'gelu3_in': [], 'ln4_in': [], 'gelu4_in': []}
    
    def hook_fn(name):
        def hook(module, input, output):
            ranges[name].append((input[0].min().item(), input[0].max().item()))
        return hook
        
    hooks = [
        model.ln1.register_forward_hook(hook_fn('ln1_in')),
        model.gelu1.register_forward_hook(hook_fn('gelu1_in')),
        model.ln2.register_forward_hook(hook_fn('ln2_in')),
        model.gelu2.register_forward_hook(hook_fn('gelu2_in')),
        
        model.ln3.register_forward_hook(hook_fn('ln3_in')),
        model.gelu3.register_forward_hook(hook_fn('gelu3_in')),
        model.ln4.register_forward_hook(hook_fn('ln4_in')),
        model.gelu4.register_forward_hook(hook_fn('gelu4_in')),
    ]
    
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(loader):
            if i >= num_batches: break
            model(batch_x.to(device))
            
    for h in hooks: h.remove()
    
    final_scales = {}
    for k, v in ranges.items():
        min_v = min([x[0] for x in v])
        max_v = max([x[1] for x in v])
        final_scales[k] = max(abs(min_v), abs(max_v))
    return final_scales


# ─────────────────────────────────────────────────────────────────────────────
# 4. Theoretical Energy Savings Calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_vae_energy(in_dim, hidden_dim1, hidden_dim2, latent_dim, snn_model, approx_linear, approx_ln, approx_gelu):
    # Total baseline ANN dynamic energy
    # Standard MAC energy = 4.6 pJ
    macs_encoder = (in_dim * hidden_dim1) + (hidden_dim1 * hidden_dim2) + (hidden_dim2 * latent_dim) * 2
    macs_decoder = (latent_dim * hidden_dim2) + (hidden_dim2 * hidden_dim1) + (hidden_dim1 * in_dim)
    total_macs = macs_encoder + macs_decoder
    e_ann_linear = total_macs * 4.6
    
    e_ann_ln = 2 * (14.7 * hidden_dim1 + 41.8) + 2 * (14.7 * hidden_dim2 + 41.8)
    e_ann_gelu = 2 * (hidden_dim1 * 65.4) + 2 * (hidden_dim2 * 65.4)
    # final sigmoid
    e_ann_sigmoid = in_dim * 65.4
    e_ann_total = e_ann_linear + e_ann_ln + e_ann_gelu + e_ann_sigmoid
    
    # Target SNN VAE energy
    if approx_linear:
        e_snn_linear = total_macs * (0.57 + 0.9)
    else:
        e_snn_linear = e_ann_linear
        
    if approx_ln:
        ln1_sq = snn_model.ln1.total_sq_spikes / max(snn_model.ln1.num_elements, 1)
        ln1_v = snn_model.ln1.total_v_spikes / max(snn_model.ln1.num_samples, 1)
        ln2_sq = snn_model.ln2.total_sq_spikes / max(snn_model.ln2.num_elements, 1)
        ln2_v = snn_model.ln2.total_v_spikes / max(snn_model.ln2.num_samples, 1)
        ln3_sq = snn_model.ln3.total_sq_spikes / max(snn_model.ln3.num_elements, 1)
        ln3_v = snn_model.ln3.total_v_spikes / max(snn_model.ln3.num_samples, 1)
        ln4_sq = snn_model.ln4.total_sq_spikes / max(snn_model.ln4.num_elements, 1)
        ln4_v = snn_model.ln4.total_v_spikes / max(snn_model.ln4.num_samples, 1)
        
        e_ln1 = (ln1_sq * 0.9 + (ln1_v * 0.9) / hidden_dim1 + 1.47) * hidden_dim1
        e_ln2 = (ln2_sq * 0.9 + (ln2_v * 0.9) / hidden_dim2 + 1.47) * hidden_dim2
        e_ln3 = (ln3_sq * 0.9 + (ln3_v * 0.9) / hidden_dim2 + 1.47) * hidden_dim2
        e_ln4 = (ln4_sq * 0.9 + (ln4_v * 0.9) / hidden_dim1 + 1.47) * hidden_dim1
        e_snn_ln = e_ln1 + e_ln2 + e_ln3 + e_ln4
    else:
        e_snn_ln = e_ann_ln
        
    if approx_gelu:
        gelu1_spikes = snn_model.gelu1.total_spikes / max(snn_model.gelu1.num_elements, 1)
        gelu2_spikes = snn_model.gelu2.total_spikes / max(snn_model.gelu2.num_elements, 1)
        gelu3_spikes = snn_model.gelu3.total_spikes / max(snn_model.gelu3.num_elements, 1)
        gelu4_spikes = snn_model.gelu4.total_spikes / max(snn_model.gelu4.num_elements, 1)
        
        e_snn_gelu = ((gelu1_spikes * 0.1) * hidden_dim1 + (gelu2_spikes * 0.1) * hidden_dim2 +
                      (gelu3_spikes * 0.1) * hidden_dim2 + (gelu4_spikes * 0.1) * hidden_dim1)
    else:
        e_snn_gelu = e_ann_gelu
        
    e_snn_total = e_snn_linear + e_snn_ln + e_snn_gelu + e_ann_sigmoid
    cdcer = (1.0 - e_snn_total / e_ann_total) * 100.0
    
    return e_ann_total, e_snn_total, cdcer


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Comparative Script Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generative VAE Image SNN Conversion using Mitchell C-2 & S-PLA")
    parser.add_argument('--timesteps', '-T', type=int, default=16, help='SNN Timesteps T')
    parser.add_argument('--prefix_k', '-k', type=int, default=3, help='S-PLA prefix spikes K')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Executing on device: {device}")
    
    plot_dir = os.path.join(current_dir, "plots", "mitchell_c2_snn")
    model_dir = os.path.join(current_dir, "test_model")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load Fashion-MNIST Data
    train_loader, test_loader = generate_fashion_data(batch_size=args.batch_size)
    
    # 2. Check if a standard VAE is already pre-trained and parse structure dynamically
    model_path = os.path.join(model_dir, "fashion_vae.pth")
    if os.path.exists(model_path):
        print(f"\nPre-trained VAE found at {model_path}. Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load hyperparameters dynamically from trained metadata
        hidden_dim1 = checkpoint.get('hidden_dim1', 512)
        hidden_dim2 = checkpoint.get('hidden_dim2', 256)
        latent_dim = checkpoint.get('latent_dim', 20)
        
        vae_ann = ToyVAE(
            input_dim=784, 
            hidden_dim1=hidden_dim1, 
            hidden_dim2=hidden_dim2, 
            latent_dim=latent_dim
        ).to(device)
        vae_ann.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"\n[Error] Pre-trained VAE checkpoint not found at {model_path}. Run train_vae.py first.")
        return
        
    vae_ann.eval()
    
    # 3. Dynamic Activation Calibration
    print("\nProfiling activation ranges on training subset for optimal SNN scale mapping...")
    scales = calibrate_vae_ranges(vae_ann, train_loader, device, num_batches=10)
    
    pad = 1.15
    s_ln1 = round(scales['ln1_in'] * pad, 2)
    s_ln2 = round(scales['ln2_in'] * pad, 2)
    s_ln3 = round(scales['ln3_in'] * pad, 2)
    s_ln4 = round(scales['ln4_in'] * pad, 2)
    
    s_gelu1 = round(scales['gelu1_in'] * pad, 2)
    s_gelu2 = round(scales['gelu2_in'] * pad, 2)
    s_gelu3 = round(scales['gelu3_in'] * pad, 2)
    s_gelu4 = round(scales['gelu4_in'] * pad, 2)
    
    print(f"  - Dynamic LayerNorm Scales: ln1={s_ln1}, ln2={s_ln2}, ln3={s_ln3}, ln4={s_ln4}")
    print(f"  - Dynamic GELU Activ Scales: gelu1={s_gelu1}, gelu2={s_gelu2}, gelu3={s_gelu3}, gelu4={s_gelu4}")
    
    # 4. SNN VAE Conversion
    print("\nApplying Training-Free SNN VAE Conversion (Deeper Hierarchy)...")
    vae_snn = SpikingVAE(
        ann_model=vae_ann,
        timesteps=args.timesteps,
        s_ln1=s_ln1, s_ln2=s_ln2, s_ln3=s_ln3, s_ln4=s_ln4,
        s_gelu1=s_gelu1, s_gelu2=s_gelu2, s_gelu3=s_gelu3, s_gelu4=s_gelu4,
        prefix_k=args.prefix_k,
        approx_linear=True,
        approx_ln=True,
        approx_gelu=True
    ).to(device)
    vae_snn.eval()
    
    # 5. Evaluate Reconstruction MSE on test loader
    print("\nEvaluating Image Reconstruction Quality (MSE) on Test Set...")
    ann_mse = 0.0
    snn_mse = 0.0
    total_samples = 0
    
    # Reset SNN Layer metrics
    for ln in [vae_snn.ln1, vae_snn.ln2, vae_snn.ln3, vae_snn.ln4]:
        ln.total_sq_spikes = 0.0
        ln.total_v_spikes = 0.0
        ln.num_elements = 0
        ln.num_samples = 0
    for gelu in [vae_snn.gelu1, vae_snn.gelu2, vae_snn.gelu3, vae_snn.gelu4]:
        gelu.total_spikes = 0.0
        gelu.num_elements = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_ann, _, _ = vae_ann(data)
            recon_snn, _, _ = vae_snn(data)
            
            ann_mse += F.mse_loss(recon_ann, data, reduction='sum').item()
            snn_mse += F.mse_loss(recon_snn, data, reduction='sum').item()
            total_samples += data.size(0)
            
    ann_avg_mse = ann_mse / (total_samples * 784)
    snn_avg_mse = snn_mse / (total_samples * 784)
    print(f"  - ANN Base Reconstruction MSE : {ann_avg_mse:.5f}")
    print(f"  - Spiking SNN Reconstruction MSE: {snn_avg_mse:.5f} (Delta: {snn_avg_mse - ann_avg_mse:+.5f})")
    
    # 6. Generate Reconstructions Plot
    print("\nGenerating Reconstruction Comparison Plot...")
    data, _ = next(iter(test_loader))
    data = data[:10].to(device)
    with torch.no_grad():
        recon_ann, _, _ = vae_ann(data)
        recon_snn, _, _ = vae_snn(data)
        
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        # Original
        axes[0, i].imshow(data[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original", loc='left', weight='bold')
        
        # ANN Recon
        axes[1, i].imshow(recon_ann[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("ANN Reconstructed", loc='left', weight='bold')
        
        # SNN Recon
        axes[2, i].imshow(recon_snn[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title(f"SNN (T={args.timesteps}) Reconstructed", loc='left', weight='bold')
        
    plt.suptitle("Fashion-MNIST VAE Image Reconstruction: Original vs. ANN vs. Spiking SNN", fontsize=14, weight='bold')
    recon_plot_path = os.path.join(plot_dir, "vae_reconstruction_report.png")
    plt.savefig(recon_plot_path, dpi=150, bbox_inches='tight')
    print(f"  [Success] Reconstruction comparative report saved to: {recon_plot_path}")
    
    # 7. Brand New Image Generation from Latent Space Noise!
    print("\nGenerating BRAND NEW images from random noise z ~ N(0, I) through SNN Decoder...")
    with torch.no_grad():
        # Sample 10 random latent vectors
        z = torch.randn(10, latent_dim).to(device)
        gen_ann = vae_ann.decode(z)
        gen_snn = vae_snn.decode(z)
        
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.8))
    for i in range(10):
        # ANN generated
        axes[0, i].imshow(gen_ann[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("ANN Generated", loc='left', weight='bold')
        
        # SNN generated
        axes[1, i].imshow(gen_snn[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"SNN (T={args.timesteps}) Generated", loc='left', weight='bold')
        
    plt.suptitle("Generative Test: Brand New Fashion-MNIST Images generated from latent noise z ~ N(0, I)", fontsize=14, weight='bold')
    gen_plot_path = os.path.join(plot_dir, "vae_generation_report.png")
    plt.savefig(gen_plot_path, dpi=150, bbox_inches='tight')
    print(f"  [Success] Generative comparative report saved to: {gen_plot_path}")
    
    # 8. Theoretical Dynamic Energy Analysis
    e_ann, e_snn, savings = calculate_vae_energy(
        784, hidden_dim1, hidden_dim2, latent_dim, vae_snn,
        approx_linear=True, approx_ln=True, approx_gelu=True
    )
    
    print("\n" + "="*90)
    print("HYBRID EXPERIMENT COMPILATION REPORT (Fashion-MNIST VAE - GENERATIVE DOMAIN)")
    print("="*90)
    print(f"  - ANN Base Reconstruction MSE: {ann_avg_mse:.5f}")
    print(f"  - SNN Spiking Reconstruction MSE: {snn_avg_mse:.5f} (MSE Delta: {snn_avg_mse - ann_avg_mse:+.5f})")
    print(f"  - SNN Firing rate LN1 Sq Spikes: {vae_snn.ln1.total_sq_spikes / max(vae_snn.ln1.num_elements, 1):.2f} spikes/element")
    print(f"  - SNN Firing rate LN2 Sq Spikes: {vae_snn.ln2.total_sq_spikes / max(vae_snn.ln2.num_elements, 1):.2f} spikes/element")
    print(f"  - SNN Firing rate LN3 Sq Spikes: {vae_snn.ln3.total_sq_spikes / max(vae_snn.ln3.num_elements, 1):.2f} spikes/element")
    print(f"  - SNN Firing rate LN4 Sq Spikes: {vae_snn.ln4.total_sq_spikes / max(vae_snn.ln4.num_elements, 1):.2f} spikes/element")
    print(f"  - SNN Firing rate GELU1 Spikes  : {vae_snn.gelu1.total_spikes / max(vae_snn.gelu1.num_elements, 1):.2f} spikes")
    print(f"  - SNN Firing rate GELU2 Spikes  : {vae_snn.gelu2.total_spikes / max(vae_snn.gelu2.num_elements, 1):.2f} spikes")
    print(f"  - SNN Firing rate GELU3 Spikes  : {vae_snn.gelu3.total_spikes / max(vae_snn.gelu3.num_elements, 1):.2f} spikes")
    print(f"  - SNN Firing rate GELU4 Spikes  : {vae_snn.gelu4.total_spikes / max(vae_snn.gelu4.num_elements, 1):.2f} spikes")
    print("-" * 90)
    print(f"  - ANN Total Dynamic Energy   : {e_ann / 1e6:.4f} uJ per sample")
    print(f"  - SNN Total Dynamic Energy   : {e_snn / 1e6:.4f} uJ per sample")
    print(f"  - Realized Generative Savings: {savings:.2f}% ({e_ann / e_snn:.2f}x lower energy!)")
    print("="*90)
    print()

if __name__ == "__main__":
    main()
