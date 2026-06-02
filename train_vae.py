"""
Train Fashion-MNIST VAE for SNN Conversion
==========================================
Trains a standard PyTorch Variational Autoencoder (VAE) on the Fashion-MNIST dataset
and saves the checkpoint to `test_model/fashion_vae.pth` for spiking SNN evaluation.
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure local imports are on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_utils import ToyVAE, get_device


def loss_function(recon_x, x, mu, logvar):
    # Binary Cross Entropy reconstruction loss + KL Divergence regularization loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


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


def main():
    parser = argparse.ArgumentParser(description="Train standard VAE on Fashion-MNIST")
    parser.add_argument('--hidden_dim1', type=int, default=512, help='First hidden layer dimension')
    parser.add_argument('--hidden_dim2', type=int, default=256, help='Second hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_name', type=str, default='fashion_vae.pth', help='Filename to save VAE weights')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load data
    train_loader, test_loader = generate_fashion_data(batch_size=args.batch_size)
    
    # 2. Initialize VAE model (Deeper Multi-layer Architecture)
    model = ToyVAE(
        input_dim=784, 
        hidden_dim1=args.hidden_dim1, 
        hidden_dim2=args.hidden_dim2, 
        latent_dim=args.latent_dim
    ).to(device)
    
    # 3. Train VAE
    print(f"\nTraining Fashion-MNIST VAE: Hidden1={args.hidden_dim1}, Hidden2={args.hidden_dim2}, Latent={args.latent_dim}, Epochs={args.epochs}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"  - Epoch {epoch+1}/{args.epochs} | Avg Loss: {train_loss / len(train_loader.dataset):.4f}")
        
    # 4. Save trained model
    save_dir = os.path.join(current_dir, 'test_model')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.save_name)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'input_dim': 784,
        'hidden_dim1': args.hidden_dim1,
        'hidden_dim2': args.hidden_dim2,
        'latent_dim': args.latent_dim,
        'data_type': 'Fashion-MNIST'
    }
    
    torch.save(checkpoint, save_path)
    print(f"\nFashion-MNIST VAE Training complete. Model saved to {save_path}\n")
    
    # 5. Generate and Save 3-sample test reconstruction comparative plot
    print("Generating training evaluation comparative plot (3 samples)...")
    model.eval()
    plot_dir = os.path.join(current_dir, 'plots', 'mitchell_c2_snn')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get 3 test samples
    test_iter = iter(test_loader)
    data, _ = next(test_iter)
    samples = data[:3].to(device)
    
    with torch.no_grad():
        recon_samples, _, _ = model(samples)
        
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i in range(3):
        # Original Fashion Image
        axes[0, i].imshow(samples[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Sample {i+1} Original")
        
        # Reconstructed Fashion Image
        axes[1, i].imshow(recon_samples[i].cpu().view(28, 28).numpy(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Sample {i+1} Reconstructed")
        
    plt.suptitle("ANN VAE Fashion-MNIST Reconstruction Quality (3 Test Samples)", fontsize=12, weight='bold')
    fig_path = os.path.join(plot_dir, 'fashion_vae_train_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[Success] Reconstruction comparative plot saved to: {fig_path}\n")


if __name__ == "__main__":
    main()
