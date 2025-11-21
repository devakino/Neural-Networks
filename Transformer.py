import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# Device configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Simple Transformer for multiplying by 3
# ----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=16, num_heads=2, d_ff=64, num_layers=2, seq_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input embedding
        self.embed = nn.Linear(1, d_model)
        # Positional encoding
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.out = nn.Linear(d_model, 1)
        
    def forward(self, src, tgt):
        # src, tgt: (batch, seq_len, 1)
        src = self.embed(src) + self.pos
        tgt = self.embed(tgt) + self.pos
        
        # Transformer expects shape (seq_len, batch, d_model)
        src = src.transpose(0,1)
        tgt = tgt.transpose(0,1)
        
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        
        out = self.out(out).transpose(0,1)  # back to (batch, seq_len, 1)
        return out

# ----------------------------
# Training setup
# ----------------------------
torch.manual_seed(42)

seq_len = 4
batch_size = 128
n_samples = 5000

# Generate random input integers 0-9
X = torch.randint(0, 10, (n_samples, seq_len, 1), dtype=torch.float).to(device)
Y = 3 * X  # output is input * 3

# Create shifted target input for decoder (teacher forcing)
tgt_input = torch.zeros_like(Y)
tgt_input[:,1:,:] = Y[:,:-1,:]
tgt_input = tgt_input.to(device)

# Model, loss, optimizer
model = SimpleTransformer(d_model=64, num_heads=2, d_ff=64, num_layers=4, seq_len=seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(3000):
    optimizer.zero_grad()
    out = model(X, tgt_input)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# Testing
# ----------------------------
test_X = torch.randint(0, 10, (5, seq_len, 1), dtype=torch.float).to(device)
test_tgt = torch.zeros_like(test_X).to(device)
test_out = model(test_X, test_tgt).detach()

print("\nInput Sequences:")
print(test_X.squeeze(-1).cpu())
print("Predicted Sequences (should be x*3):")
print(test_out.squeeze(-1).cpu())
print("Expected Sequences:")
print((3*test_X.squeeze(-1)).cpu())
