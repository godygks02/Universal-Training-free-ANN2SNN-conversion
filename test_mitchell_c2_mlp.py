"""
MNIST MLP Floating-Point Multiplication Approximation using Mitchell C-2 & S-PLA
=============================================================================
Loads the pre-trained MNIST MLP and replaces standard components with:
1. Linear Layers: Mitchell's Logarithmic Multiplier with 2-Term LUT-based Segmented Correction (Mitchell C-2, 0.57 pJ)
2. LayerNorm: IEEE 754 Exponent-Guided Bit-Slice S-PLA (0.0 pJ encoder + Sign-Controlled Accumulator)
3. GELU Activation: BFE Spike-Driven S-PLA Activation (Shift-and-Add, 0.1 pJ addition)

Evaluates classification accuracy and analyzes total dynamic energy (pJ per sample).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_utils import generate_data
from model_utils import ToyTransformerMLP, get_device, calculate_ann_energy

# Import modularized components
from modules.IEEE_754_based_Encoding import IEEE754_based_encoder
import importlib
mitchell_c2_approx = importlib.import_module("modules.mitchell_c-2_approx")
spla_module = importlib.import_module("modules.S-PLA")

decompose_float32 = mitchell_c2_approx.decompose_float32
mitchell_c2_multiply = mitchell_c2_approx.mitchell_c2_multiply
MitchellC2Linear = mitchell_c2_approx.MitchellC2Linear
SBTSPLAActivation = spla_module.SBTSPLAActivation
SPLALayerNorm = spla_module.SPLALayerNorm
SPLAActivationWrapper = spla_module.SPLAActivationWrapper


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mitchell C-2 & S-PLA Hybrid MLP Model
# ─────────────────────────────────────────────────────────────────────────────

class MitchellSPLATransformerMLP(nn.Module):
    def __init__(self, ann_model, timesteps=16, s_ln1=3.0, s_ln2=3.0, s_gelu1=4.0, s_gelu2=4.0, prefix_k=3,
                 approx_linear=True, approx_ln=True, approx_gelu=True):
        super().__init__()
        
        # fc1: 784 -> 256
        if approx_linear:
            self.fc1 = MitchellC2Linear(ann_model.fc1.in_features, ann_model.fc1.out_features)
            self.fc1.load_from_standard_linear(ann_model.fc1)
        else:
            self.fc1 = nn.Linear(ann_model.fc1.in_features, ann_model.fc1.out_features)
            with torch.no_grad():
                self.fc1.weight.copy_(ann_model.fc1.weight)
                if ann_model.fc1.bias is not None:
                    self.fc1.bias.copy_(ann_model.fc1.bias)
                    
        # ln1: SPLALayerNorm or standard LayerNorm
        if approx_ln:
            self.ln1 = SPLALayerNorm(ann_model.ln1.normalized_shape[0], timesteps=timesteps, s_val=s_ln1)
            self.ln1.load_from_standard_layernorm(ann_model.ln1)
        else:
            self.ln1 = nn.LayerNorm(ann_model.ln1.normalized_shape[0])
            with torch.no_grad():
                self.ln1.weight.copy_(ann_model.ln1.weight)
                self.ln1.bias.copy_(ann_model.ln1.bias)
                
        # gelu1: SPLAActivationWrapper or standard GELU
        if approx_gelu:
            self.gelu1 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu1, prefix_k=prefix_k)
        else:
            self.gelu1 = nn.GELU()
            
        # fc2: 256 -> 256
        if approx_linear:
            self.fc2 = MitchellC2Linear(ann_model.fc2.in_features, ann_model.fc2.out_features)
            self.fc2.load_from_standard_linear(ann_model.fc2)
        else:
            self.fc2 = nn.Linear(ann_model.fc2.in_features, ann_model.fc2.out_features)
            with torch.no_grad():
                self.fc2.weight.copy_(ann_model.fc2.weight)
                if ann_model.fc2.bias is not None:
                    self.fc2.bias.copy_(ann_model.fc2.bias)
                    
        # ln2:
        if approx_ln:
            self.ln2 = SPLALayerNorm(ann_model.ln2.normalized_shape[0], timesteps=timesteps, s_val=s_ln2)
            self.ln2.load_from_standard_layernorm(ann_model.ln2)
        else:
            self.ln2 = nn.LayerNorm(ann_model.ln2.normalized_shape[0])
            with torch.no_grad():
                self.ln2.weight.copy_(ann_model.ln2.weight)
                self.ln2.bias.copy_(ann_model.ln2.bias)
                
        # gelu2:
        if approx_gelu:
            self.gelu2 = SPLAActivationWrapper(target_name='gelu', timesteps=timesteps, scale_factor=s_gelu2, prefix_k=prefix_k)
        else:
            self.gelu2 = nn.GELU()
            
        # fc3: 256 -> 10
        if approx_linear:
            self.fc3 = MitchellC2Linear(ann_model.fc3.in_features, ann_model.fc3.out_features)
            self.fc3.load_from_standard_linear(ann_model.fc3)
        else:
            self.fc3 = nn.Linear(ann_model.fc3.in_features, ann_model.fc3.out_features)
            with torch.no_grad():
                self.fc3.weight.copy_(ann_model.fc3.weight)
                if ann_model.fc3.bias is not None:
                    self.fc3.bias.copy_(ann_model.fc3.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu2(x)
        logits = self.fc3(x)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation Loop and Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_proposed_model(model, loader, device):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # Reset trackers (only for approximated layers to avoid AttributeError)
    for m in [model.ln1, model.ln2]:
        if hasattr(m, 'total_sq_spikes'):
            m.total_sq_spikes = 0.0
            m.total_v_spikes = 0.0
            m.num_elements = 0
            m.num_samples = 0
        
    for m in [model.gelu1, model.gelu2]:
        if hasattr(m, 'total_spikes'):
            m.total_spikes = 0.0
            m.num_elements = 0
        
    print(f"Evaluating Mitchell C-2 + S-PLA Hybrid SNN on {len(loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    acc = 100.0 * correct / total
    return acc


def calibrate_ranges(model, loader, device):
    model.eval()
    ranges = {'fc1_in': [], 'fc1_w': [], 'ln1_in': [], 'gelu1_in': [], 
              'fc2_in': [], 'fc2_w': [], 'ln2_in': [], 'gelu2_in': [], 
              'fc3_in': [], 'fc3_w': []}
    
    ranges['fc1_w'] = (model.fc1.weight.min().item(), model.fc1.weight.max().item())
    ranges['fc2_w'] = (model.fc2.weight.min().item(), model.fc2.weight.max().item())
    ranges['fc3_w'] = (model.fc3.weight.min().item(), model.fc3.weight.max().item())

    def hook_fn(name):
        def hook(module, input, output):
            ranges[name].append((input[0].min().item(), input[0].max().item()))
        return hook

    hooks = [
        model.fc1.register_forward_hook(hook_fn('fc1_in')),
        model.ln1.register_forward_hook(hook_fn('ln1_in')),
        model.gelu1.register_forward_hook(hook_fn('gelu1_in')),
        model.fc2.register_forward_hook(hook_fn('fc2_in')),
        model.ln2.register_forward_hook(hook_fn('ln2_in')),
        model.gelu2.register_forward_hook(hook_fn('gelu2_in')),
        model.fc3.register_forward_hook(hook_fn('fc3_in')),
    ]

    with torch.no_grad():
        for batch_x, _ in loader:
            model(batch_x.to(device))
    for h in hooks: h.remove()
    
    final_ranges = {}
    for k, v in ranges.items():
        if isinstance(v, tuple):
            final_ranges[k] = (round(v[0], 2), round(v[1], 2))
        else:
            final_ranges[k] = (round(min([x[0] for x in v]), 2), round(max([x[1] for x in v]), 2))
    return final_ranges


def main():
    parser = argparse.ArgumentParser(description="Test MNIST MLP with Mitchell C-2 Multiplier and S-PLA activations")
    parser.add_argument('--load_name', type=str, default='mnist_mlp.pth', help='Checkpoint name to load')
    parser.add_argument('--timesteps', '-T', type=int, default=16, help='Number of SNN timesteps T')
    parser.add_argument('--prefix_k', '-k', type=int, default=3, help='S-PLA spike prefix routing bits K')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--no_fp_mul', action='store_true', help='Disable Mitchell C-2 weight multiplication (use standard float)')
    parser.add_argument('--no_norm', action='store_true', help='Disable S-PLA LayerNorm (use standard LayerNorm)')
    parser.add_argument('--no_act', action='store_true', help='Disable S-PLA GELU activations (use standard GELU)')
    args = parser.parse_args()
    
    device = get_device()
    
    # Paths setup
    exp_dir = os.path.join(current_dir, "test_model")
    plot_dir = os.path.join(current_dir, "plots", "mitchell_c2_snn")
    os.makedirs(plot_dir, exist_ok=True)
    
    ann_path = os.path.join(exp_dir, args.load_name)
    if not os.path.exists(ann_path):
        # Fallback to relative path search if absolute path is not found
        fallback_path = os.path.join("test_model", args.load_name)
        if os.path.exists(fallback_path):
            ann_path = fallback_path
        else:
            print(f"[Error] Pre-trained model not found at {ann_path} or {fallback_path}. Run train_ann.py first.")
            return
        
    # 1. Load Trained ANN
    print(f"Loading ANN checkpoint from {ann_path}...")
    checkpoint = torch.load(ann_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_classes = checkpoint['num_classes']
    
    ann_model = ToyTransformerMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    ann_model.load_state_dict(checkpoint['state_dict'])
    ann_model.eval()
    
    # Load MNIST Data
    train_loader, test_loader = generate_data(batch_size=args.batch_size)
    
    # Evaluate Standard ANN accuracy
    ann_correct = 0
    ann_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = ann_model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            ann_total += batch_y.size(0)
            ann_correct += (predicted == batch_y).sum().item()
    ann_acc = 100.0 * ann_correct / ann_total
    print(f"  - ANN Base Test Accuracy: {ann_acc:.2f}%")
    
    # 2. Dynamic Activation Range Profiling (Zero-Clipping)
    print("\nInvestigating activation ranges on MNIST train set...")
    final_ranges = calibrate_ranges(ann_model, train_loader, device)
    
    s_ln1 = max(abs(final_ranges['ln1_in'][0]), abs(final_ranges['ln1_in'][1]))
    s_gelu1 = max(abs(final_ranges['gelu1_in'][0]), abs(final_ranges['gelu1_in'][1]))
    s_ln2 = max(abs(final_ranges['ln2_in'][0]), abs(final_ranges['ln2_in'][1]))
    s_gelu2 = max(abs(final_ranges['gelu2_in'][0]), abs(final_ranges['gelu2_in'][1]))
    
    # Pad ranges slightly to cover any unseen test out-of-distribution values
    s_ln1 = round(s_ln1 * 1.15, 2)
    s_gelu1 = round(s_gelu1 * 1.15, 2)
    s_ln2 = round(s_ln2 * 1.15, 2)
    s_gelu2 = round(s_gelu2 * 1.15, 2)
    
    print(f"  - Profiled Scale Factors: s_ln1={s_ln1}, s_gelu1={s_gelu1}, s_ln2={s_ln2}, s_gelu2={s_gelu2}")

    # 3. Instantiate and Evaluate Proposed Mitchell C-2 + S-PLA Hybrid Model
    approx_linear = not args.no_fp_mul
    approx_ln = not args.no_norm
    approx_gelu = not args.no_act
    
    print("\nConverting model to Mitchell C-2 + S-PLA Hybrid Spiking MLP...")
    print(f"  - Approximation settings: Linear={approx_linear}, LayerNorm={approx_ln}, GELU={approx_gelu}")
    
    hybrid_model = MitchellSPLATransformerMLP(
        ann_model=ann_model,
        timesteps=args.timesteps,
        s_ln1=s_ln1,
        s_ln2=s_ln2,
        s_gelu1=s_gelu1,
        s_gelu2=s_gelu2,
        prefix_k=args.prefix_k,
        approx_linear=approx_linear,
        approx_ln=approx_ln,
        approx_gelu=approx_gelu
    ).to(device)
    
    snn_acc = evaluate_proposed_model(hybrid_model, test_loader, device)
    
    # 4. Energy Calculations & Metrics Compiler
    # Calculate Standard ANN Energy Baseline
    ann_energy_breakdown = calculate_ann_energy(input_dim, hidden_dim, num_classes)
    e_ann_total = ann_energy_breakdown['total_pj']
    e_ann_linear = ann_energy_breakdown['breakdown_pj']['linear']
    e_ann_ln = ann_energy_breakdown['breakdown_pj']['ln']
    e_ann_gelu = ann_energy_breakdown['breakdown_pj']['gelu']
    
    # Calculate Proposed Mitchell C-2 + S-PLA Hybrid SNN Energy (pJ per sample)
    
    # A. Linear Layers Energy (Mitchell C-2 @ 0.57 pJ + Standard FPU Addition @ 0.9 pJ for summing products)
    if approx_linear:
        num_weights = (input_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
        num_biases = hidden_dim + hidden_dim + num_classes
        # Mitchell C-2 multiplication = 0.57 pJ, Standard FPU addition = 0.9 pJ
        e_snn_linear = (num_weights * (0.57 + 0.9)) + (num_biases * 0.9)
    else:
        e_snn_linear = e_ann_linear
        
    # B. LayerNorms Energy (S-PLA + Mitchell C-2 Hybrid Energy model: S-PLA Square + S-PLA InvSqrt + Mitchell C-2)
    if approx_ln:
        ln1_sq_spikes = hybrid_model.ln1.total_sq_spikes / max(hybrid_model.ln1.num_elements, 1)
        ln1_v_spikes = hybrid_model.ln1.total_v_spikes / max(hybrid_model.ln1.num_samples, 1)
        ln2_sq_spikes = hybrid_model.ln2.total_sq_spikes / max(hybrid_model.ln2.num_elements, 1)
        ln2_v_spikes = hybrid_model.ln2.total_v_spikes / max(hybrid_model.ln2.num_samples, 1)
        
        # Energy per element:
        # e_sq = T * sq_fr * 0.9 pJ = ln_sq_spikes * 0.9 pJ
        # e_inv = (T * v_fr * 0.9) / D pJ = (ln_v_spikes * 0.9) / hidden_dim
        # e_mult = 1.47 pJ (Mitchell C-2)
        e_ln1_per_element = (ln1_sq_spikes * 0.9) + ((ln1_v_spikes * 0.9) / hidden_dim) + 1.47
        e_ln2_per_element = (ln2_sq_spikes * 0.9) + ((ln2_v_spikes * 0.9) / hidden_dim) + 1.47
        
        e_snn_ln1 = e_ln1_per_element * hidden_dim
        e_snn_ln2 = e_ln2_per_element * hidden_dim
        e_snn_ln = e_snn_ln1 + e_snn_ln2
    else:
        ln1_sq_spikes = 0.0
        ln2_sq_spikes = 0.0
        e_snn_ln = e_ann_ln
        
    # C. GELU Activations Energy (S-PLA shift-and-add: spikes * 0.1 pJ, or Standard GELU @ 65.4 pJ per element)
    if approx_gelu:
        gelu1_spikes = hybrid_model.gelu1.total_spikes / max(hybrid_model.gelu1.num_elements, 1)
        gelu2_spikes = hybrid_model.gelu2.total_spikes / max(hybrid_model.gelu2.num_elements, 1)
        e_snn_gelu1 = (gelu1_spikes * 0.1) * hidden_dim
        e_snn_gelu2 = (gelu2_spikes * 0.1) * hidden_dim
        e_snn_gelu = e_snn_gelu1 + e_snn_gelu2
    else:
        gelu1_spikes = 0.0
        gelu2_spikes = 0.0
        e_snn_gelu = e_ann_gelu
        
    # D. Total SNN Energy (including standard Softmax energy as it remains unapproximated float in this model)
    e_ann_softmax = ann_energy_breakdown['breakdown_pj']['softmax']
    e_snn_total = e_snn_linear + e_snn_ln + e_snn_gelu + e_ann_softmax
    
    # Efficiency Gain (CDCER)
    cdcer = (1.0 - e_snn_total / e_ann_total) * 100.0
    
    print("\n" + "="*100)
    print("HYBRID EXPERIMENT COMPILATION REPORT (MNIST MLP)")
    print("="*100)
    print(f"  - ANN Base Accuracy          : {ann_acc:.2f}%")
    print(f"  - Proposed SNN Accuracy      : {snn_acc:.2f}% (Accuracy Drop: {ann_acc - snn_acc:.2f}%)")
    if approx_ln:
        print(f"  - S-PLA LayerNorm1 Sq Spikes : {ln1_sq_spikes:.2f} spikes/element")
        print(f"  - S-PLA LayerNorm2 Sq Spikes : {ln2_sq_spikes:.2f} spikes/element")
    else:
        print("  - LayerNorm1 & 2             : Standard Float (No Approximation)")
    if approx_gelu:
        print(f"  - S-PLA GELU1 Spikes/Steps   : {gelu1_spikes:.2f} spikes")
        print(f"  - S-PLA GELU2 Spikes/Steps   : {gelu2_spikes:.2f} spikes")
    else:
        print("  - GELU1 & 2                  : Standard Float (No Approximation)")
    print("-" * 100)
    print(f"  - ANN Total Dynamic Energy   : {e_ann_total/1e6:.4f} uJ")
    print(f"  - SNN Total Dynamic Energy   : {e_snn_total/1e6:.4f} uJ")
    print(f"  - Realized Energy Savings    : {cdcer:.2f}% ({e_ann_total/e_snn_total:.2f}x lower energy!)")
    print("="*100)
    
    # 4. Save visual report table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table_data = [
        ["Metric", "ANN (Baseline)", f"Proposed SNN (T={args.timesteps}, K={args.prefix_k})", "Efficiency Gain / Delta"],
        ["Accuracy", f"{ann_acc:.2f}%", f"{snn_acc:.2f}%", f"{ann_acc - snn_acc:.2f}% (Drop)"],
        ["Linear Layers Energy", f"{e_ann_linear/1e6:.3f} uJ", f"{e_snn_linear/1e6:.3f} uJ", f"{(1 - e_snn_linear/e_ann_linear)*100:.1f}% Savings"],
        ["LayerNorm Energy", f"{e_ann_ln/1e6:.3f} uJ", f"{e_snn_ln/1e6:.3f} uJ", f"{(1 - e_snn_ln/e_ann_ln)*100:.1f}% Savings"],
        ["GELU Activation Energy", f"{e_ann_gelu/1e6:.3f} uJ", f"{e_snn_gelu/1e6:.3f} uJ", f"{(1 - e_snn_gelu/e_ann_gelu)*100:.1f}% Savings"],
        ["Total Dynamic Energy", f"{e_ann_total/1e6:.3f} uJ", f"{e_snn_total/1e6:.3f} uJ", f"{cdcer:.2f}% (Savings)"],
        ["Inference energy reduction", "1.0x (Base)", f"{e_ann_total/e_snn_total:.2f}x Lower", "-"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e3f2fd') # Blue accent header
            
    plt.title(f"Mitchell C-2 & S-PLA Hybrid Spiking MLP Verification Report\n(MNIST MLP, T={args.timesteps})", pad=20, weight='bold', color='#1A237E')
    save_path = os.path.join(plot_dir, "mitchell_c2_snn_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Success] Numerical report successfully saved to:\n  {save_path}\n")

if __name__ == "__main__":
    main()
