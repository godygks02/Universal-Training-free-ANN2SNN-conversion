import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ToyTransformerMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gelu1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.gelu2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu2(x)
        logits = self.fc3(x)
        return logits

def train_ann(model, train_loader, device, epochs=10, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Self-contained evaluation for training accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")

def calculate_ann_energy(in_dim, hidden_dim, num_classes):
    """
    Calculate theoretical ANN energy consumption based on hardware analysis PDF.
    Returns energy in pJ (pico-Joules) for a single forward pass.
    """
    # 1. Linear Layers (MACs * 4.6 pJ)
    # FC1: in_dim -> hidden_dim
    # FC2: hidden_dim -> hidden_dim
    # FC3: hidden_dim -> num_classes
    total_macs = (in_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
    e_linear = total_macs * 4.6
    
    # 2. LayerNorm (2 layers of size hidden_dim)
    # E_LN(n) = 14.7 * n + 41.8 pJ
    e_ln = 2 * (14.7 * hidden_dim + 41.8)
    
    # 3. GELU Activation (2 layers of size hidden_dim)
    # E_GELU = 65.4 pJ per element
    e_gelu = 2 * (hidden_dim * 65.4)
    
    # 4. Softmax (1 layer of size num_classes)
    # E_Softmax(n) = 58.0 * n - 0.9 pJ
    e_softmax = 58.0 * num_classes - 0.9
    
    total_energy = e_linear + e_ln + e_gelu + e_softmax
    
    return {
        'total_pj': total_energy,
        'total_uj': total_energy / 1e6,
        'breakdown_pj': {
            'linear': e_linear,
            'ln': e_ln,
            'gelu': e_gelu,
            'softmax': e_softmax
        }
    }

class ToyVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=512, hidden_dim2=256, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        
        # Encoder Pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.gelu1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.gelu2 = nn.GELU()
        
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        
        # Decoder Pathway
        self.fc3 = nn.Linear(latent_dim, hidden_dim2)
        self.ln3 = nn.LayerNorm(hidden_dim2)
        self.gelu3 = nn.GELU()
        
        self.fc4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.ln4 = nn.LayerNorm(hidden_dim1)
        self.gelu4 = nn.GELU()
        
        self.fc5 = nn.Linear(hidden_dim1, input_dim)
        
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
