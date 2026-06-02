import torch
import os
import argparse
from data_utils import generate_data
from model_utils import ToyTransformerMLP, train_ann, get_device, calculate_ann_energy

def main():
    parser = argparse.ArgumentParser(description="Train ANN MNIST MLP for SNN conversion")
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Save/Path arguments
    parser.add_argument('--save_name', type=str, default='mnist_mlp.pth', help='Filename to save the model')
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # MNIST fixed parameters
    N_FEATURES = 784
    N_CLASSES = 10
    
    # 1. Load MNIST data
    train_loader, test_loader = generate_data(batch_size=args.batch_size)
    
    # 2. Initialize large MLP model
    model = ToyTransformerMLP(
        input_dim=N_FEATURES,
        hidden_dim=args.hidden_dim,
        num_classes=N_CLASSES
    ).to(device)
    
    # 3. Train model
    print(f"Training MNIST MLP: Hidden={args.hidden_dim}, Epochs={args.epochs}, Batch={args.batch_size}")
    train_ann(model, train_loader, device, epochs=args.epochs, lr=args.lr)
    
    # 4. Save the trained model with metadata
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, 'test_model')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.save_name)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'input_dim': N_FEATURES,
        'hidden_dim': args.hidden_dim,
        'num_classes': N_CLASSES,
        'data_type': 'MNIST' # Tag for consistency in test.py
    }
    
    torch.save(checkpoint, save_path)
    print(f"\nMNIST ANN Training complete. Model saved to {save_path}")

    # 5. Theoretical Energy Analysis
    energy = calculate_ann_energy(N_FEATURES, args.hidden_dim, N_CLASSES)
    print("\n" + "="*40)
    print("Theoretical ANN Energy Consumption (Per Sample)")
    print("-" * 40)
    print(f"Linear Ops:   {energy['breakdown_pj']['linear']/1e3:>10.2f} nJ")
    print(f"LayerNorm:    {energy['breakdown_pj']['ln']/1e3:>10.2f} nJ")
    print(f"GELU:         {energy['breakdown_pj']['gelu']/1e3:>10.2f} nJ")
    print(f"Softmax:      {energy['breakdown_pj']['softmax']/1e3:>10.2f} nJ")
    print("-" * 40)
    print(f"Total Energy: {energy['total_uj']:>10.4f} uJ")
    print("="*40)

if __name__ == "__main__":
    main()
