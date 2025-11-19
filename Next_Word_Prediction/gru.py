# -------------------------------------------------------------
# 🧠 Email Next-Word Prediction using Vanilla GRU (PyTorch)
# -------------------------------------------------------------
import torch                                                            # For PyTorch functionalities
import torch.nn as nn                                                   # For getting the neural network layer
import torch.optim as optim                                             # For getting the optimizer
from torch.utils.data import DataLoader, TensorDataset                  # For creating tensors of the data and loading them in batches
import numpy as np
import os

# -------------------------------------------------------------
#   Load dataset
# -------------------------------------------------------------
DATA_PATH = "Next_Word_Prediction/data/data.txt"

# Checking if the dataset exists or not and if it exists then reading data from it
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

with open(DATA_PATH, "r") as f:
    sentences = [line.strip().lower() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(sentences)} sentences from {DATA_PATH}")

# -------------------------------------------------------------
#   Preprocessing - tokenization + mapping
# -------------------------------------------------------------

# Creating the dictionary (Dictionary will contain the words and the indices of the words)
words = set(" ".join(sentences).split())
word_to_idx = {word: i for i, word in enumerate(words)}                 # Defining all unique words by an unique index
idx_to_word = {i: word for word, i in word_to_idx.items()}              # Same as above but in reverse format
vocab_size = len(words)                                                 # Number of unique words define the vocabulary size

# Now from the formed dictionary we change the character data to numerical input.
def sentence_to_indices(sentence):
    return [word_to_idx[w] for w in sentence.split() if w in word_to_idx]

# Create input→target pairs for every sentence
# All words before the current word are the input and the word itself is the target.
X, Y = [], []
for s in sentences:
    tokens = sentence_to_indices(s)
    for i in range(1, len(tokens)):
        X.append(tokens[:i])                                            # All words before current word are its input
        Y.append(tokens[i])                                             # The currennt word is the output(target).

# Now the different tokens inside of the inputs have different sizes so we standardize them using the zero padding method.
# Pad sequences (right-aligned)
max_len = max(len(x) for x in X)
X_padded = np.zeros((len(X), max_len), dtype=int)
for i, seq in enumerate(X):
    X_padded[i, -len(seq):] = seq

# Now using the PyTorch functionality of ".tensor" we create the tensors of the data (For the Padded inputs and the Target outputs)
X_tensor = torch.tensor(X_padded, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# Now we create a paired dataset using the TensorDataset and the Dataloader will load those pairs randomly in batches of 16
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------------------------------------
# 3️⃣ Define the GRU model
# -------------------------------------------------------------
class EmailGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):            # Using specific hyperparameters
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)           # Creating the embeddings of the data as the input layer
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)    # Creating the GRU layer
        self.fc = nn.Linear(hidden_size, vocab_size)                    # Last Fully-Connected Layer for the output layer
    
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)                                      # Recurrence using the hidden state
        out, hidden = self.gru(embeds, hidden)
        out = self.fc(out[:, -1, :])  # ✅ use last hidden state
        return out, hidden

# -------------------------------------------------------------
# 4️⃣ Initialize model, loss, optimizer
# -------------------------------------------------------------
embed_size = 128                                                        # Hyperparameters
hidden_size = 128
model = EmailGRU(vocab_size, embed_size, hidden_size)                   # Defining the model

criterion = nn.CrossEntropyLoss()                                       # Loss calculator
optimizer = optim.Adam(model.parameters(), lr=0.001)                    # Optimizer

# -------------------------------------------------------------
# 5️⃣ Train the model
# -------------------------------------------------------------
epochs = 1000                                                           # Training Iterations
print("\n🚀 Training started...\n")
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:                                     # For every 16 batch size data we compute the backpropogation
        optimizer.zero_grad()                                           # Initializing the optimizer 
        output, _ = model(batch_x)                                      # Generating the output for an input
        loss = criterion(output, batch_y)                               # Caculating the loss by comparing the generated output with the target output
        loss.backward()                                                 # Back-Propogation
        optimizer.step()                                                # Using the optimizer for back-propogation
        total_loss += loss.item()                                       # Total loss is the summation of the losses per iteration
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

print("\n✅ Training complete!")

# -------------------------------------------------------------
# Prediction function
# -------------------------------------------------------------
def predict_next_word(model, text, hidden=None, top_k=3):
    model.eval()                                                        # Changes the model from training mode to evaluation mode
    with torch.no_grad():                                               # Diasbles the gradient computation
        tokens = sentence_to_indices(text.lower())                      # For the entered sentence/word, creates indices using the dictionary and also checks if the entered word exist in the dictionary or not
        if not tokens:
            return ["⚠️ Unknown words"], hidden
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)         # Create tensor for the entered input
        output, hidden = model(x, hidden)                               # Handling the ouput and the hidden state for further words suggestions
        probs = torch.softmax(output, dim=1)                            # Using softmax to create probabilities
        top_idx = torch.topk(probs, top_k).indices.squeeze(0).tolist()  # Returning the 3 most relevant results
        predictions = [idx_to_word[i] for i in top_idx]                 # Returning the words according to the index
    return predictions, hidden

# -------------------------------------------------------------
# 7️⃣ Interactive next-word prediction with context memory
# -------------------------------------------------------------
print("\n📨 Email Smart Compose System (GRU) 🧠")
print("Type a few words and get next word suggestions!")
print("Type 'exit' to quit.\n")

context_words = []
hidden_state = None

while True:
    text = input("You: ").strip().lower()
    if text == "exit":
        print("👋 Goodbye!")
        break
    context_words += text.split()
    prompt = " ".join(context_words[-10:])  # last 10 words as context
    suggestions, hidden_state = predict_next_word(model, prompt, hidden_state)
    # detach hidden state to prevent backprop through entire history
    if hidden_state is not None:
        hidden_state = hidden_state.detach()
    print("Suggestions →", suggestions)
