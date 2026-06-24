"""
Segment Anything Model (SAM-ViT-B) SNN Conversion & Benchmark with Full Attention & Softmax Approximation
======================================================================================================
Converts facebook/sam-vit-base model utilizing:
1. Linear Layers: Mitchell's Logarithmic Multiplier (0.57 pJ mult + 0.9 pJ add = 1.47 pJ per MAC)
2. LayerNorm: Hybrid S-PLA Square (Mitchell C-2 squaring) + S-PLA InvSqrt + Mitchell C-2 Dynamic Scaling
3. GELU Activation: BFE Spike-Driven S-PLA Activation (Pure Shift-and-Add)
4. Attention Score Multiplications (QK^T and AV): Mitchell C-2 Logarithmic Matrix Multiplication (1.47 pJ per MAC)
5. Attention Softmax: Proposed IEEE 754 Exponent-Guided Bit-Slice S-PLA Softmax (Pure Shift-and-Add)

Evaluates image segmentation performance (IoU / boundary precision) on a synthetic image and tracks energy efficiency.
"""

import os
import sys
import torch
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor
import torch.nn as nn
import importlib

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import modularized components
from modules.IEEE_754_based_Encoding import IEEE754_based_encoder
mitchell_c2_approx = importlib.import_module("modules.mitchell_c-2_approx")
spla_module = importlib.import_module("modules.S-PLA")

decompose_float32 = mitchell_c2_approx.decompose_float32
mitchell_c2_multiply = mitchell_c2_approx.mitchell_c2_multiply
MitchellC2Linear = mitchell_c2_approx.MitchellC2Linear
mitchell_c2_matmul_qk = mitchell_c2_approx.mitchell_c2_matmul_qk
mitchell_c2_matmul_av = mitchell_c2_approx.mitchell_c2_matmul_av
ProposedSoftmaxSPLA = spla_module.ProposedSoftmaxSPLA
SPLALayerNorm = spla_module.SPLALayerNorm
SPLAActivationWrapper = spla_module.SPLAActivationWrapper


# ─────────────────────────────────────────────────────────────────────────────
# 1. Custom Mitchell C-2 SamVisionAttention Replacement
# ─────────────────────────────────────────────────────────────────────────────

class MitchellC2SamVisionAttention(nn.Module):
    """
    Mitchell C-2 & S-PLA full-approx replacement for SAM's SamVisionSdpaAttention / SamVisionAttention.
    """
    def __init__(self, original_attn, timesteps=16, s_softmax=8.0, lut=None):
        super().__init__()
        self.num_attention_heads = original_attn.num_attention_heads
        self.scale = original_attn.scale
        
        self.use_rel_pos = original_attn.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = original_attn.rel_pos_h
            self.rel_pos_w = original_attn.rel_pos_w
            self.get_decomposed_rel_pos = original_attn.get_decomposed_rel_pos
            
        self.qkv = MitchellC2Linear(
            original_attn.qkv.in_features, 
            original_attn.qkv.out_features, 
            bias=(original_attn.qkv.bias is not None)
        )
        self.qkv.load_from_standard_linear(original_attn.qkv)
        
        self.proj = MitchellC2Linear(
            original_attn.proj.in_features, 
            original_attn.proj.out_features, 
            bias=(original_attn.proj.bias is not None)
        )
        self.proj.load_from_standard_linear(original_attn.proj)
        
        if lut is None:
            lut = torch.tensor([
                [0.0156, 0.0469, 0.0781, 0.1094],
                [0.0469, 0.1406, 0.2344, 0.3281],
                [0.0781, 0.2344, 0.3906, 0.5469],
                [0.1094, 0.3281, 0.5469, 0.7656]
            ], dtype=torch.float32)
        self.register_buffer('lut', lut)
        
        self.attn_softmax = ProposedSoftmaxSPLA(timesteps=timesteps, s_val=s_softmax)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, height, width, _ = hidden_states.shape
        
        # 1. Mitchell C-2 qkv projection
        qkv_out = self.qkv(hidden_states)
        
        qkv = (
            qkv_out
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_bias = None
        if self.use_rel_pos:
            decomposed_rel_pos = self.get_decomposed_rel_pos(
                query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )
            decomposed_rel_pos = decomposed_rel_pos.reshape(
                batch_size, self.num_attention_heads, height * width, height * width
            )
            attn_bias = decomposed_rel_pos

        query = query.view(batch_size, self.num_attention_heads, height * width, -1)
        key = key.view(batch_size, self.num_attention_heads, height * width, -1)
        value = value.view(batch_size, self.num_attention_heads, height * width, -1)

        # Scale query
        query = query * self.scale
        
        # 2. Mitchell C-2 QK^T matrix multiplication
        attn_weights = mitchell_c2_matmul_qk(query, key.transpose(-1, -2), self.lut)
        
        if attn_bias is not None:
            # Apply attention bias (relative position embeddings)
            attn_weights = attn_weights + attn_bias
            
        # 3. Proposed Exponent-Guided S-PLA Softmax
        attn_weights_softmax = self.attn_softmax(attn_weights, dim=-1)
        
        # 4. Mitchell C-2 AV matrix multiplication
        attn_output = mitchell_c2_matmul_av(attn_weights_softmax, value, self.lut)
        
        # 5. Reshape and project
        attn_output = (
            attn_output.view(batch_size, self.num_attention_heads, height, width, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch_size, height, width, -1)
        )

        attn_output = self.proj(attn_output)
        return attn_output, None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def download_coco_val_images():
    """Downloads sample COCO validation images from images.cocodataset.org."""
    import requests
    import io
    
    samples = [
        {
            "url": "http://images.cocodataset.org/val2017/000000039769.jpg",  # Two cats
            "point": [[[345, 230]]],  # Point on one of the cats
            "name": "Cats"
        },
        {
            "url": "http://images.cocodataset.org/val2017/000000000285.jpg",  # Bear
            "point": [[[200, 240]]],  # Point on the bear
            "name": "Bear"
        },
        {
            "url": "http://images.cocodataset.org/val2017/000000000632.jpg",  # Sheep
            "point": [[[320, 240]]],  # Point on one of the sheep
            "name": "Sheep"
        }
    ]
    
    loaded_samples = []
    print("\nDownloading sample COCO validation images...")
    for s in samples:
        try:
            resp = requests.get(s["url"], timeout=10)
            if resp.status_code == 200:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                loaded_samples.append({
                    "image": img,
                    "point": s["point"],
                    "name": s["name"]
                })
                print(f"  Successfully downloaded: {s['name']}")
        except Exception as e:
            print(f"  Failed to download {s['name']}: {e}")
            
    if not loaded_samples:
        print("  Warning: No internet or download failed. Falling back to synthetic image.")
        img = Image.new("RGB", (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.ellipse([150, 150, 362, 362], fill=(230, 50, 50))
        loaded_samples.append({
            "image": img,
            "point": [[[256, 256]]],
            "name": "SyntheticCircle"
        })
        
    return loaded_samples



# ─────────────────────────────────────────────────────────────────────────────
# 3. Calibration and Replacement Logic
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_sam(model, inputs, device):
    """Calibrates activations in SAM's heavy vision encoder."""
    model.eval()
    ranges = {}

    def get_hook(name):
        def hook(module, input, output):
            val = input[0].detach()
            val_abs = val.abs()
            val_flat = val_abs.reshape(-1).float()
            if val_flat.numel() > 1000000:
                stride = val_flat.numel() // 1000000
                val_flat = val_flat[::stride][:1000000]
            q_999 = torch.quantile(val_flat, 0.999).item()
            if name not in ranges:
                ranges[name] = [-q_999, q_999]
            else:
                ranges[name][1] = max(ranges[name][1], q_999)
                ranges[name][0] = -ranges[name][1]
        return hook

    hooks = []
    layers = model.vision_encoder.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.layer_norm1.register_forward_hook(get_hook(f"layer_{i}_ln_before")))
        hooks.append(layer.layer_norm2.register_forward_hook(get_hook(f"layer_{i}_ln_after")))
        hooks.append(layer.mlp.act.register_forward_hook(get_hook(f"layer_{i}_gelu")))

    print("\nCalibrating SAM vision encoder activations...")
    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return ranges


def replace_sam_modules_with_approx(model, ranges, args, device):
    """Substitutes linear projections, LayerNorms, GELUs, and attentions in SAM's image encoder."""
    print(f"\nReplacing Vision Encoder modules (Linear={args.replace_linear}, LN={args.replace_ln}, GELU={args.replace_gelu}, Attention MatMul & Softmax={args.replace_attn})...")
    pad = 1.15
    lut_data = torch.tensor([
        [0.0156, 0.0469, 0.0781, 0.1094],
        [0.0469, 0.1406, 0.2344, 0.3281],
        [0.0781, 0.2344, 0.3906, 0.5469],
        [0.1094, 0.3281, 0.5469, 0.7656]
    ], dtype=torch.float32)

    layers = model.vision_encoder.layers

    for i, layer in enumerate(layers):
        # A. LayerNorms
        if args.replace_ln:
            r_before = ranges[f"layer_{i}_ln_before"]
            scale_before = max(abs(r_before[0]), abs(r_before[1])) * pad
            mbe_ln_b = SPLALayerNorm(layer.layer_norm1.normalized_shape[0], timesteps=args.timesteps, s_val=scale_before).to(device)
            mbe_ln_b.load_from_standard_layernorm(layer.layer_norm1)
            layer.layer_norm1 = mbe_ln_b

            r_after = ranges[f"layer_{i}_ln_after"]
            scale_after = max(abs(r_after[0]), abs(r_after[1])) * pad
            mbe_ln_a = SPLALayerNorm(layer.layer_norm2.normalized_shape[0], timesteps=args.timesteps, s_val=scale_after, approx_square='mitchell').to(device)
            mbe_ln_a.load_from_standard_layernorm(layer.layer_norm2)
            layer.layer_norm2 = mbe_ln_a

        # B. GELU Activation in MLP
        if args.replace_gelu:
            r_gelu = ranges[f"layer_{i}_gelu"]
            scale_gelu = max(abs(r_gelu[0]), abs(r_gelu[1])) * pad
            mbe_gelu = SPLAActivationWrapper(target_name='gelu', timesteps=args.timesteps, scale_factor=scale_gelu, prefix_k=args.prefix_k).to(device)
            layer.mlp.act = mbe_gelu

        # C. Attention score calculations & Softmax S-PLA
        if args.replace_attn:
            scale_softmax = 8.0
            approx_attn = MitchellC2SamVisionAttention(layer.attn, timesteps=args.timesteps, s_softmax=scale_softmax, lut=lut_data)
            layer.attn = approx_attn.to(device)

        # D. MLP Linear Layers (lin1, lin2)
        if args.replace_linear:
            if not isinstance(layer.mlp.lin1, MitchellC2Linear):
                mbe_lin1 = MitchellC2Linear(layer.mlp.lin1.in_features, layer.mlp.lin1.out_features, bias=(layer.mlp.lin1.bias is not None)).to(device)
                mbe_lin1.load_from_standard_linear(layer.mlp.lin1)
                layer.mlp.lin1 = mbe_lin1
            if not isinstance(layer.mlp.lin2, MitchellC2Linear):
                mbe_lin2 = MitchellC2Linear(layer.mlp.lin2.in_features, layer.mlp.lin2.out_features, bias=(layer.mlp.lin2.bias is not None)).to(device)
                mbe_lin2.load_from_standard_linear(layer.mlp.lin2)
                layer.mlp.lin2 = mbe_lin2

    print("SNN SAM Vision Encoder Conversion Complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Energy compilation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sam_ann_energy(seq_len=4096):
    """Calculates theoretical baseline ANN energy for SAM-ViT-B vision encoder per sample."""
    D = 768
    num_layers = 12
    
    # Attention QKV and Out Projection + MLP FC1 and FC2
    macs_attn_qkv = 3 * D * D
    macs_attn_proj = D * D
    macs_mlp_fc = 4 * D * D
    macs_mlp_proj = 4 * D * D
    macs_attn_scores = 2 * seq_len * D # QK^T and AV
    macs_layer = macs_attn_qkv + macs_attn_proj + macs_mlp_fc + macs_mlp_proj + macs_attn_scores
    
    macs_total_linear = num_layers * macs_layer * seq_len
    
    e_linear = macs_total_linear * 4.6 # pJ (MAC = mult + add = 3.7 + 0.9 = 4.6 pJ)
    
    e_ln = seq_len * (2 * num_layers) * (14.7 * D + 41.8)
    e_gelu = seq_len * num_layers * (4 * D * 65.4)
    e_softmax = num_layers * 12 * seq_len * (58.0 * seq_len - 0.9)
    
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


def calculate_snn_energy(approx_model, args, seq_len=4096):
    """Estimates SNN energy consumption using active spike tracking."""
    D = 768
    num_layers = 12
    
    ann_energy_breakdown = calculate_sam_ann_energy(seq_len=seq_len)
    e_ann_total = ann_energy_breakdown['total_pj']
    e_ann_linear = ann_energy_breakdown['breakdown_pj']['linear']
    e_ann_ln = ann_energy_breakdown['breakdown_pj']['ln']
    e_ann_gelu = ann_energy_breakdown['breakdown_pj']['gelu']
    e_ann_softmax = ann_energy_breakdown['breakdown_pj']['softmax']

    # A. Linear Layers Energy
    if args.replace_linear:
        num_weights_layers = 12 * (3 * D * D + D * D + 4 * D * D + 4 * D * D)
        e_linear_approx = num_weights_layers * (0.57 + 0.9) * seq_len
        
        macs_attn_scores = num_layers * 2 * seq_len * D * seq_len
        if args.replace_attn:
            e_linear_attn_scores = macs_attn_scores * (0.57 + 0.9)
        else:
            e_linear_attn_scores = macs_attn_scores * 4.6
            
        e_snn_linear = e_linear_approx + e_linear_attn_scores
    else:
        e_snn_linear = e_ann_linear

    # B. LayerNorms Energy
    if args.replace_ln:
        layers = approx_model.vision_encoder.layers
        e_snn_ln = 0.0
        ln_sq_spikes_list = []
        for layer in layers:
            # ln1
            sq_spikes = layer.layer_norm1.total_sq_spikes / max(layer.layer_norm1.num_elements, 1)
            v_spikes = layer.layer_norm1.total_v_spikes / max(layer.layer_norm1.num_samples, 1)
            ln_sq_spikes_list.append(sq_spikes)
            e_snn_ln += (1.47 + (v_spikes * 0.9) / D + 1.47) * D * seq_len
            
            # ln2
            sq_spikes = layer.layer_norm2.total_sq_spikes / max(layer.layer_norm2.num_elements, 1)
            v_spikes = layer.layer_norm2.total_v_spikes / max(layer.layer_norm2.num_samples, 1)
            ln_sq_spikes_list.append(sq_spikes)
            e_snn_ln += (1.47 + (v_spikes * 0.9) / D + 1.47) * D * seq_len
            
        avg_ln_sq_spikes = sum(ln_sq_spikes_list) / len(ln_sq_spikes_list) if len(ln_sq_spikes_list) > 0 else 0.0
    else:
        avg_ln_sq_spikes = 0.0
        e_snn_ln = e_ann_ln

    # C. GELU Activations Energy
    if args.replace_gelu:
        layers = approx_model.vision_encoder.layers
        e_snn_gelu = 0.0
        gelu_spikes_list = []
        for layer in layers:
            act_fn = layer.mlp.act
            if isinstance(act_fn, SPLAActivationWrapper):
                spikes = act_fn.total_spikes / max(act_fn.num_elements, 1)
                gelu_spikes_list.append(spikes)
                e_snn_gelu += (spikes * 0.1) * (4 * D) * seq_len
                
        avg_gelu_spikes = sum(gelu_spikes_list) / len(gelu_spikes_list) if len(gelu_spikes_list) > 0 else 0.0
    else:
        avg_gelu_spikes = 0.0
        e_snn_gelu = e_ann_gelu

    # D. Softmax SNN Energy
    if args.replace_attn:
        e_snn_softmax = 0.0
        attn_spikes_list = []
        for layer in approx_model.vision_encoder.layers:
            if isinstance(layer.attn, MitchellC2SamVisionAttention):
                m = layer.attn.attn_softmax
                spikes = m.total_spikes / max(m.num_elements, 1)
                attn_spikes_list.append(spikes)
                # SAM has 12 heads
                e_attn_softmax_block = (spikes * 1.0) * (12 * seq_len * seq_len)
                e_snn_softmax += e_attn_softmax_block
                
        avg_attn_spikes = sum(attn_spikes_list) / len(attn_spikes_list) if len(attn_spikes_list) > 0 else 0.0
    else:
        avg_attn_spikes = 0.0
        e_snn_softmax = e_ann_softmax

    e_snn_total = e_snn_linear + e_snn_ln + e_snn_gelu + e_snn_softmax
    return e_ann_total, e_snn_total, e_ann_linear, e_snn_linear, e_ann_ln, e_snn_ln, e_ann_gelu, e_snn_gelu, e_ann_softmax, e_snn_softmax, avg_ln_sq_spikes, avg_gelu_spikes, avg_attn_spikes


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def reset_trackers(model):
    """Resets active firing rate monitors inside SNN layers."""
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


def main():
    parser = argparse.ArgumentParser(description="SAM-ViT-B Spiking Conversion & Boundary-preserved Segmentation Verification")
    parser.add_argument('--model_id', type=str, default='facebook/sam-vit-base', help='HuggingFace SAM Model ID')
    parser.add_argument('--timesteps', '-T', type=int, default=16, help='Encoding timesteps T')
    parser.add_argument('--prefix_k', '-k', type=int, default=3, help='S-PLA routing bits K')
    
    # Ablation flags
    parser.add_argument('--no_fp_mul', action='store_true', help='Disable Mitchell C-2 projections')
    parser.add_argument('--no_norm', action='store_true', help='Disable S-PLA LayerNorm')
    parser.add_argument('--no_act', action='store_true', help='Disable S-PLA GELU activations')
    parser.add_argument('--no_attn', action='store_true', help='Disable Mitchell C-2 Attention & S-PLA Softmax')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SAM model and processor
    print("Loading SAM-ViT-B model & processor...")
    processor = SamProcessor.from_pretrained(args.model_id)
    model = SamModel.from_pretrained(args.model_id).to(device)
    model.eval()

    # 2. Download/Load COCO val samples
    loaded_samples = download_coco_val_images()
    
    # Enable approximate modules
    args.replace_linear = not args.no_fp_mul
    args.replace_ln = not args.no_norm
    args.replace_gelu = not args.no_act
    args.replace_attn = not args.no_attn

    ious = []
    fidelities = []
    vis_data = None

    # 3. Process each real-world sample
    for idx, sample in enumerate(loaded_samples):
        raw_image = sample["image"]
        input_points = sample["point"]
        name = sample["name"]
        
        inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)

        # Baseline ANN Inference
        with torch.no_grad():
            outputs_ann = model(**inputs)
            
        masks_ann = processor.image_processor.post_process_masks(
            outputs_ann.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )[0][0, 0].numpy()

        # Calibration
        calibration_ranges = calibrate_sam(model, inputs, device)

        # Create fresh SNN model for this configuration
        snn_model = SamModel.from_pretrained(args.model_id).to(device)
        snn_model.eval()
        snn_model = replace_sam_modules_with_approx(snn_model, calibration_ranges, args, device)

        # Perform SNN inference
        reset_trackers(snn_model)
        with torch.no_grad():
            outputs_snn = snn_model(**inputs)
            
        masks_snn = processor.image_processor.post_process_masks(
            outputs_snn.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )[0][0, 0].numpy()

        # Evaluate metrics
        intersection = np.logical_and(masks_ann, masks_snn).sum()
        union = np.logical_or(masks_ann, masks_snn).sum()
        iou = intersection / max(union, 1)
        
        diff_mask = np.logical_xor(masks_ann, masks_snn)
        boundary_diff_pixels = diff_mask.sum()
        total_pixels = masks_ann.size
        boundary_fidelity = (1.0 - boundary_diff_pixels / total_pixels) * 100.0

        ious.append(iou)
        fidelities.append(boundary_fidelity)
        
        print(f"\nSample {idx+1} ({name}) Metrics:")
        print(f"  - ANN vs SNN Mask IoU            : {iou * 100:.3f}%")
        print(f"  - Boundary Fidelity (Pixel Match): {boundary_fidelity:.4f}%")
        print(f"  - Discrepancy Pixels            : {boundary_diff_pixels} out of {total_pixels} px")

        # Save sample for visualization
        if vis_data is None:
            vis_data = []
        vis_data.append({
            "raw_image": raw_image,
            "input_points": input_points,
            "masks_ann": masks_ann,
            "masks_snn": masks_snn,
            "diff_mask": diff_mask,
            "iou": iou,
            "fidelity": boundary_fidelity,
            "name": name
        })

    # Print average metrics
    avg_iou = np.mean(ious)
    avg_fidelity = np.mean(fidelities)
    print("\n" + "="*80)
    print("SAM-VIT-B SPIKING CONVERSION METRIC REPORT (AVERAGE)")
    print("="*80)
    print(f"  - Average ANN vs SNN Mask IoU   : {avg_iou * 100:.3f}%")
    print(f"  - Average Boundary Fidelity    : {avg_fidelity:.4f}%")
    print("="*80)

    # 4. Plot Results for all samples
    if vis_data:
        num_samples = len(vis_data)
        fig, axes = plt.subplots(num_samples, 4, figsize=(18, 4.5 * num_samples))
        
        # Ensure axes is 2D even if there's only 1 sample
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for r, vis in enumerate(vis_data):
            # Column 0: Original image + Prompt
            axes[r, 0].imshow(vis["raw_image"])
            pt = vis["input_points"][0][0]
            axes[r, 0].plot(pt[0], pt[1], 'go', markersize=10, label="Prompt Point")
            axes[r, 0].set_title(f"{vis['name']} (Original)", weight='bold')
            axes[r, 0].axis('off')
            if r == 0:
                axes[r, 0].legend()
                
            # Column 1: ANN Mask
            axes[r, 1].imshow(vis["masks_ann"], cmap='gray')
            axes[r, 1].set_title("ANN Base Mask", weight='bold', color='#1A237E')
            axes[r, 1].axis('off')
            
            # Column 2: SNN Mask
            axes[r, 2].imshow(vis["masks_snn"], cmap='gray')
            axes[r, 2].set_title(f"SNN Mask (IoU: {vis['iou']*100:.2f}%)", weight='bold', color='#2E7D32')
            axes[r, 2].axis('off')
            
            # Column 3: Deviation Map
            diff_vis = np.zeros((*vis["masks_ann"].shape, 3), dtype=np.uint8)
            diff_vis[vis["masks_ann"]] = [100, 100, 200]
            diff_vis[vis["diff_mask"]] = [255, 0, 0]
            axes[r, 3].imshow(diff_vis)
            axes[r, 3].set_title(f"Deviation Map (Fidelity: {vis['fidelity']:.3f}%)", weight='bold', color='#D84315')
            axes[r, 3].axis('off')

        plt.suptitle("Segment Anything Model (SAM-ViT-B) SNN Conversion & Boundary Analysis on COCO Images", fontsize=16, weight='bold', y=0.98)
        plt.tight_layout()
        
        plot_dir = "plots/mitchell_c2_snn"
        os.makedirs(plot_dir, exist_ok=True)
        visual_save_path = os.path.join(plot_dir, "sam_vitb_segmentation_comparison.png")
        plt.savefig(visual_save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n[Success] Segmentation comparison image successfully saved to:\n  {visual_save_path}\n")


if __name__ == "__main__":
    main()
