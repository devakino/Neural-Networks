# -------------------------------------------------------------
# Email Next-Word Prediction with Context Memory (PyTorch)
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
DATA_PATH = "Next_Word_Prediction/data/data.txt"

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
#   Defining the RNN Model
# -------------------------------------------------------------
class EmailRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        out = self.fc(out[:, -1, :])  # use last time step
        return out, hidden

# -------------------------------------------------------------
#   Hyperparameters
# -------------------------------------------------------------
embed_size = 128
hidden_size = 128
model = EmailRNN(vocab_size, embed_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------------------
#   Training
# -------------------------------------------------------------
epochs = 1000
print("\n🚀 Training started...\n")
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

print("\n✅ Training complete!")

# -------------------------------------------------------------
#   Prediction with memory (hidden state)
# -------------------------------------------------------------
def predict_next_word(model, text, hidden=None, top_k=3):
    model.eval()
    with torch.no_grad():
        tokens = sentence_to_indices(text.lower())
        if not tokens:
            return ["⚠️ Unknown words"], hidden
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        embeds = model.embedding(x)
        out, hidden = model.rnn(embeds, hidden)
        out = model.fc(out[:, -1, :])
        probs = torch.softmax(out, dim=1)
        top_idx = torch.topk(probs, top_k).indices.squeeze(0).tolist()
        predictions = [idx_to_word[i] for i in top_idx]
    return predictions, hidden

# -------------------------------------------------------------
#   Interactive Chat Loop
# -------------------------------------------------------------
print("\n📨 Email Smart Compose System 🧠")
print("Type a few words and get next word suggestions!")
print("Type 'exit' to quit or 'reset' to clear context.\n")

hidden = None  # Initialize hidden state

while True:
    text = input("You: ").strip().lower()
    if text == "exit":
        print("👋 Goodbye!")
        break
    elif text == "reset":
        hidden = None
        print("🔄 Context reset.")
        continue

    suggestions, hidden = predict_next_word(model, text, hidden)
    # Detach hidden to prevent memory from backprop graph
    if hidden is not None:
        hidden = hidden.detach()
    print("Suggestions →", suggestions)
