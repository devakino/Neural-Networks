# -------------------------------------------------------------
# Email Next-Word Prediction with Transformer (PyTorch)
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# -------------------------------------------------------------
#   Loading dataset
# -------------------------------------------------------------
DATA_PATH = "Next_Word_Prediction/data/training_data.txt"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    sentences = [line.strip().lower() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(sentences)} sentences from {DATA_PATH}")

# -------------------------------------------------------------
#   Preprocessing: Tokenization + Vocabulary
# -------------------------------------------------------------
words = set(" ".join(sentences).split())
word_to_idx = {word: i for i, word in enumerate(words)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(words)

def sentence_to_indices(sentence):
    return [word_to_idx[w] for w in sentence.split() if w in word_to_idx]

# Create input-target pairs
X, Y = [], []
for s in sentences:
    tokens = sentence_to_indices(s)
    for i in range(1, len(tokens)):
        X.append(tokens[:i])
        Y.append(tokens[i])

# Pad sequences (right-aligned)
max_len = max(len(x) for x in X)
X_padded = np.zeros((len(X), max_len), dtype=int)
for i, seq in enumerate(X):
    X_padded[i, -len(seq):] = seq

X_tensor = torch.tensor(X_padded, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------------------------------------
#   Positional Encoding
# -------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-np.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, embed_size)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# -------------------------------------------------------------
#   Transformer Model
# -------------------------------------------------------------
class EmailTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # Embed + Positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer(x)

        # Output only last position (like your RNN code)
        out = self.fc(x[:, -1, :])
        return out

# -------------------------------------------------------------
#   Hyperparameters
# -------------------------------------------------------------
embed_size = 128
num_heads = 4
hidden_dim = 256
num_layers = 2

model = EmailTransformer(vocab_size, embed_size, num_heads, hidden_dim, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------------
#   Training
# -------------------------------------------------------------
epochs = 100
print("\n🚀 Training started...\n")
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

print("\n✅ Training complete!")

# -------------------------------------------------------------
#   Prediction (Transformers do not use hidden state)
# -------------------------------------------------------------
def predict_next_word(model, text, top_k=3):
    model.eval()
    with torch.no_grad():
        tokens = sentence_to_indices(text.lower())
        if not tokens:
            return ["⚠️ Unknown words"]
        
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

        out = model(x)
        probs = torch.softmax(out, dim=1)
        top_idx = torch.topk(probs, top_k).indices.squeeze(0).tolist()
        
        predictions = [idx_to_word[i] for i in top_idx]
    return predictions

# -------------------------------------------------------------
#   Interactive Chat Loop
# -------------------------------------------------------------
print("\n📨 Email Smart Compose System 🧠")
print("Type a few words and get next word suggestions!")
print("Type 'exit' to quit.\n")

while True:
    text = input("You: ").strip().lower()
    if text == "exit":
        print("👋 Goodbye!")
        break

    suggestions = predict_next_word(model, text)
    print("Suggestions →", suggestions)
