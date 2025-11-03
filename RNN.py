import math

# Parameters of the network
Wxh = 0.3       # Weight of the hidden layer
Whh = 0.2       # Weight of the hidden state
Why = 0.1       # Weight of the output layer
bh = 0.0        # Bias of the hidden layer
by = 0.0        # Bias of the output layer
lr = 0.1       # Learning Rate

# Activation Function
def tanh(x):
    return math.tanh(x)

# Derivative of the Activation Function
def dtanh(x):
    return 1 - math.tanh(x)**2

# Input and Output
inputs = [1, 2, 1, 2]
targets = [2, 1, 2, 1]

# Training the Network
for epoch in range(200):
    h_prev = 0.0            # Storing the output for the next iteration
    hs, ys = [], []         # Lists to store the state and the output
    loss = 0.0              # Storing the loss

    # Forward pass
    for t in range(len(inputs)):    
        x = inputs[t]       # Getting the input
        h = tanh(Wxh * x + Whh * h_prev + bh)   # Output of the hidden layer/Hidden state   
        y = Why * h + by    # Ouput of the output layer
        hs.append(h)        # Storing the value to its respective list (Storing the outputs of hidden layer)
        ys.append(y)        # Storing the value to its respective list (Storing the outputs of output layer)
        h_prev = h          # Storing the current state for next iteration
        loss += 0.5 * (y - targets[t])**2       # Calculating the loss

    # Initialize gradients
    dWxh = 0.0              # Gradient for the weight of the hidden layer
    dWhh = 0.0              # Gradient for the weight of the hidden state
    dWhy = 0.0              # Gradient for the weight of the output layer
    dbh = 0.0               # Gradient for the bais of the hidden layer
    dby = 0.0               # Gradient for the bias of the output layer
    dh_next = 0.0           # Gradient for the next state

    # Backward pass (BPTT)
    for t in reversed(range(len(inputs))):
        dy = ys[t] - targets[t]             # Gradient for the output layer [Predicted - Target]
        dWhy += dy * hs[t]                  # Gradient for weights of ouput layer [Gradient for the output layer * the hidden state of the iteration]
        dby += dy                           # Gradient for bias of the output layer [equals to the graident of the output layer]

        dh = dy * Why + dh_next             # Gradient of the hidden state [Gradient of the ouput layer *  Weights of the hidden state + Gradient of the Next State]
        dh_raw = dh * dtanh(hs[t])          # Gradient of the hidden state before activation function [Gradient of the hidden state * the derivative of activation function]
        x = inputs[t]                       # Storing the input value for the iteration
        h_prev = hs[t-1] if t > 0 else 0.0  # Storing the previous state value of the iteration
        dWxh += dh_raw * x                  # Gradient of the weight of hidden layer [Gradient of the hidden state * the input for the iteration]
        dWhh += dh_raw * h_prev             # Gradient of the weight of the hidden state [Gradient of the hidden state * Previous state value]
        dbh += dh_raw                       # Gradient of the bias of the hidden layer [Gradient of the hidden state]

        dh_next = dh_raw * Whh              # Gradient of the Next state

    # Update weights (SGD)
    Wxh -= lr * dWxh                        # New value of the weight of the hidden layer
    Whh -= lr * dWhh                        # New value of the weight of the hidden state
    Why -= lr * dWhy                        # New value of the weight of the output layer
    bh -= lr * dbh                          # New value of the bias of the hidden layer
    by -= lr * dby                          # New value of the bias of the output layer

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.6f}") # Printing the loss for every 200 epochs

# Testing the RNN with the updated weights, to check the output after training for 200 epochs
h_prev = 0.0    # Iniitalzing the previous state of the network as 0
print("\nPredictions:")
for x in [1,2,1,2]:         # For our input list
    h = tanh(Wxh * x + Whh * h_prev + bh)   # Finding the hidden layer output
    y = Why * h + by                        # Finding the output layer output
    print(f"input {x} -> output {y:.3f}")   # Showcasing the input with the output
    h_prev = h                              # Assinging the value of the current state to the previous state variable that can be used by the next state
