# print("\n========== Entering the Input Layer ==========\n")
# # The input of the neural network.
# a0 = [0.5, 0.6]
# print("Input Matrix: ",a0)
# # The output of the neural network.
# y = [1.0, 1.2]

# print("\n========== Entering the Hidden Layer ==========\n")

# # Weights for the Hidden Layer.
# w1 = [
#         [0.1,   0.2],
#         [0.2,   0.4],
#         [0.3,   0.1]
#     ]
# print("Weights of the Hidden Layer: ", w1)

# # Biases for the Hidden Layer.
# b1 = [0.2,0.1,0.3]
# print("Biases of the Hidden Layer: ", b1)

# # Variable to store the summations of the multiplication of weights and input with the biases.
# z1 = [0,0,0]

# # Saving the summation of the multiplication of the weights and inputs with the baises to the variable.
# for i in range(3):
#     z1[i] = (w1[i][0]*a0[0] + w1[i][1]*a0[1]) + b1[i]

# # The output matrix before passing through the activation function.
# print("Summation of the Weight x Input + Bias:",z1)

# # Variable for storing the values after passing them through the activation function.
# act_z1 = [0,0,0] 

# # Passsing the values through the activation function.
# # Using ReLU [Only taking the positive values otherwise changing the negative values to 0]
# for i in range(3):
#     act_z1[i] = max(0, z1[i])

# # The output matrix of the hidden layer
# print("After passing through the ReLU Activation Function: ",act_z1)

# print("\n========== Entering the Output Layer ==========\n")
# # Weight matrix for the Output Layer
# w2 = [
#         [0.3,  0.1,  0.4],
#         [0.1,  0.3,  0.1]
#     ]
# print("Weights of the Output Layer: ", w2)

# # Bias matrix for the Outpout Layer
# b2 = [0.4,0.2]
# print("Biases of the Output Layer: ", b2)

# # Variable for storing the summation of the multiplication of the weights and the inputs with the biases.
# z2 = [0,0]

# # Calculation of the summation of the multiplication of the weights and the inputs with the biases.
# for i in range(2):
#     temp = 0 
#     for j in range(3):
#         temp += w2[i][j]*act_z1[j]
#     z2[i] = temp + b2[i]

# # The output matrix before passing through the activation function
# print("Summation of the Weight x Input + Bias:",z2)

# # Variable to store the outputs after passing them through the activation function
# act_z2 = [0,0]

# # Passinf the outputs through the activation function (ReLU)
# for i in range(2):
#     act_z2[i] = max(0, z2[i])

# # Final output
# o = act_z2

# # Printing the final output
# print("After passing through ReLU Activation Function",act_z2)

# # Variable to store the loss
# loss = [0,0]
# # Calculating the loss using MSE
# for i in range(2):
#     loss[i] = ((o[i]-y[i])*(o[i]-y[i]))/2

# # Printing the loss
# print("Calculated Loss: ",loss)












print("\n========== Entering the Input Layer ==========\n")

# The input of the neural network.
a0 = [0.5, 0.6]
print("Input Matrix: ",a0)

# The output of the neural network.
y = [1.0, 1.2]
print("Original Ouput Matrix: ",y)

# Weights for the Hidden Layer.
w1 = [
        [0.1,   0.2],
        [0.2,   0.4],
        [0.3,   0.1]
    ]
print("Weights of the Hidden Layer: ", w1)

# Biases for the Hidden Layer.
b1 = [0.2,0.1,0.3]
print("Biases of the Hidden Layer: ", b1)

# Variable to store the summations of the multiplication of weights and input with the biases for the hidden layer.
z1 = [0,0,0]

# Variable for storing the values after passing them through the activation function of the hidden layer.
act_z1 = [0,0,0] 

# Weight matrix for the Output Layer
w2 = [
        [0.3,  0.1,  0.4],
        [0.1,  0.3,  0.1]
    ]
print("Weights of the Output Layer: ", w2)

# Bias matrix for the Outpout Layer
b2 = [0.4,0.2]
print("Biases of the Output Layer: ", b2)

# Variable for storing the summation of the multiplication of the weights and the inputs with the biases.
z2 = [0,0]

# Variable to store the outputs after passing them through the activation function
act_z2 = [0,0]

# Variable to store the loss
loss = [0,0]

# Storing loss due the bias of Output Layer
db2 = [0,0]

# Storing loss due to the weights of the Output Layer
dw2 = [[0,0,0],[0,0,0]]

# Storing the loss due to the Output Layer
dtemp = [0,0,0]

# Storing loss due the bias of Hidden Layer
db1 = [[0,0,0],[0,0,0],[0,0,0]]



print("\n========== Entering the Hidden Layer ==========\n")

# Saving the summation of the multiplication of the weights and inputs with the baises to the variable.
for i in range(3):
    z1[i] = (w1[i][0]*a0[0] + w1[i][1]*a0[1]) + b1[i]

# The output matrix before passing through the activation function.
print("Summation of the Weight x Input + Bias:",z1)

# Passsing the values through the activation function.
# Using ReLU [Only taking the positive values otherwise changing the negative values to 0]
for i in range(3):
    act_z1[i] = max(0, z1[i])

# The output matrix of the hidden layer
print("After passing through the ReLU Activation Function: ",act_z1)

print("\n========== Entering the Output Layer ==========\n")

# Calculation of the summation of the multiplication of the weights and the inputs with the biases.
for i in range(2):
    temp = 0 
    for j in range(3):
        temp += w2[i][j]*act_z1[j]
    z2[i] = temp + b2[i]

# The output matrix before passing through the activation function
print("Summation of the Weight x Input + Bias:",z2)

# Passinf the outputs through the activation function (ReLU)
for i in range(2):
    act_z2[i] = max(0, z2[i])

# Final output
o = act_z2

# Printing the final output
print("After passing through ReLU Activation Function",act_z2)

# Calculating the loss using MSE
for i in range(2):
    loss[i] = ((o[i]-y[i])*(o[i]-y[i]))/2

# Printing the loss
print("Calculated Loss: ",loss)


print("\n========== Entering the Backpropogation ==========\n")

# Computing the error due to the bias matrix of the output layer (loss * derivative of the activation function of the Ouput Layer)
for i in range(2):
    if z2[i] > 0:
        db2[i] = loss[i]*1
    else:
        db2[i] = loss[i]*0

print("Error due to the bias for Output Layer: ",db2)

# Computing the error due to the weight matrix of the output layer (error due to bias of the output layer * Output of the previous Layer)
for i in range(2):
    for j in range(3):
        dw2[i][j] = act_z1[j]*db2[i]

print("Error due to the weights for Output Layer: ",dw2)

# Computing the error due to the Output layer (weights of the output layer * error due to the bias of the output layer)
for i in range(3):
    dtemp[i] = w2[0][i]*db2[0] + w2[1][i]*db2[1]

print("Error due to Output Layer: ",dtemp)

# Computing the error due to the Hidden layer (error due to the bias of the output layer * derivative of the activation function of the hidden layer)
for i in range(3):
    for j in range(3):
        if z1[j] > 0:
            db1[i][j] = dtemp[i] * 1
        else:
            db1[i][j] = dtemp[i] * 0

print("Error due to the ias of the Hidden Layer: ",db1)

# dw1 = 