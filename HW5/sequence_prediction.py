import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split

# Network Parameters
hiddenSize = 200
maxSequenceLength = 30
nhead = 2
num_layers = 1

# Training parameters
epochs = 200
learning_rate = 0.001

# Create a file for saving outputs
training_log_file = open('graphs/sequence_prediction/sequence_prediction_'+ str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'Sequence_prediction' + '\n')
training_log_file.flush()

print("Loading Sequence...")
# Load data
with open('./data/sequence_part1.txt', 'r', encoding='utf-8') as file:
    data = file.read()
# Gets a unique set of characters found in the sequence
# Converts the set to a list
# then it sorts the list
chars = sorted(list(set(data)))

#This line creates a dictionary that maps each character to a unique index (integer)."
ix_to_char = {i: ch for i, ch in enumerate(chars)}
#Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
char_to_ix = {ch: i for i, ch in enumerate(chars)} 
chars = sorted(list(set(data)))

inputSize = len(chars)
outputSize = len(chars)
# Save Parameters to log
training_log_file.write(f'Epochs: {epochs}, Learning Rate: {learning_rate}, Input Size: {inputSize}, Hidden Size: {hiddenSize}, Output Size: {outputSize}, Max Sequence Size: {maxSequenceLength}' + '\n')
training_log_file.flush()
training_log_file.write(f'number of heads: {nhead}, num_layers: {num_layers}' + '\n')
training_log_file.flush()

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(inputSize, hiddenSize)
        self.pos_encoder = PositionalEncoding(hiddenSize)
        encoder_layers = nn.TransformerEncoderLayer(hiddenSize, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output
    


model = SequenceModel().to(device)

# Save Number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in the model: {total_params}')
training_log_file.write('Number of parameters in model: ' + str(total_params) + '\n\n')
training_log_file.flush()

# Preparing the dataset
X = []
y = []
for i in range(len(data) - maxSequenceLength):
    sequence = data[i:i + maxSequenceLength]
    label = data[i + maxSequenceLength]
    X.append([char_to_ix[char] for char in sequence])
    y.append(char_to_ix[label])

X = np.array(X)
y = np.array(y)

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_val = torch.tensor(X_val, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# Training Tracking
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []
val_precision_list = []
val_recall_list = []
val_f1_list = []

total_time = 0

# Select loss function and the optimizer
criterion = nn.CrossEntropyLoss() # Use Cross Entropy for classification
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# Training loop
print(f'Starting training for {epochs} epochs')
for epoch in range(epochs):
    accuracy = 0.0
    start_time = time.time() # Get start time of epoch
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    accuracy = 100 * ((predicted == y_train).float().mean())

    # Calculate training accuracy for the epoch and save it
    train_accuracy_list.append(accuracy.item())
    # Save the loss from this epoch
    train_loss_list.append(loss.item()) # Save epoch loss in list

    # Output current progress of training 
    end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time # Get elapsed time of epoch
    print(f'Epoch {epoch+1} / {epochs}, Loss: {loss.item()}, Accuracy: {accuracy}%, Time: {elapsed_time} seconds')
    # Save the output to the log file
    training_log_file.write(f'Epoch {epoch+1} / {epochs}, Loss: {loss.item()}, Accuracy: {accuracy}%, Time: {elapsed_time} seconds' + '\n')
    training_log_file.flush()
    total_time += elapsed_time

    # Evaluation Loop
    print(f'Beginning validation for epoch {epoch+1}')
    start_time = time.time() # Get start time of epoch
    model.eval()

    with torch.no_grad():
        outputs = model(X_val)
        loss = criterion(outputs, y_val)
        end_time = time.time() # Get end time of epoch
        accuracy = 0.0
        _, predicted = torch.max(outputs.data, 1)
        accuracy = 100 * ((predicted == y_val).float().mean())

    # Calculate training accuracy for the epoch and save it
    val_accuracy_list.append(accuracy.item())
    # Save the loss from this epoch
    val_loss_list.append(loss.item()) # Save epoch loss in list


    elapsed_time = end_time - start_time
    total_time += elapsed_time

    print(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Loss: {loss.item()} || Accuracy: {accuracy}%')
    # Save the validation data to the log file
    training_log_file.write(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Loss: {loss.item()} || Accuracy: {accuracy}%' + '\n')
    training_log_file.flush()

print(f'Finished Training {epochs} epochs in {total_time} seconds')
# Save the data to the log file
training_log_file.write(f'Finished Training {epochs} epochs in {total_time} seconds' + '\n\n')
training_log_file.flush()

# Test new model: 
testText = '“Next character prediction is a fundamental task in the field of natural language processing (NL'
print('Test model:\n')
print(f'Input Text: {testText}\n')
print('Expected character: P\n')
initial_input = torch.tensor([char_to_ix[c] for c in testText[-maxSequenceLength:]], dtype=torch.long).unsqueeze(0).to(device)
predicted_char = model(initial_input)
predicted_index = torch.argmax(predicted_char, dim=1).item()
print(f'Predicted character: {ix_to_char[predicted_index]}\n')

# Log the Test
training_log_file.write(f'\nTesting model:\n')
training_log_file.flush()
training_log_file.write(f'Input Text: {testText}\n')
training_log_file.flush()
training_log_file.write('Expected character: P\n')
training_log_file.flush()
training_log_file.write(f'Predicted character: {ix_to_char[predicted_index]}' + '\n\n')
training_log_file.flush()

# Generate Graphs
# Training Loss Graph
plt.plot(train_loss_list, label = 'Training Loss')
plt.plot(val_loss_list, label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss VS Epoch')
plt.legend()
plt.savefig('./graphs/sequence_prediction/sequence_prediction_trainingLoss.png')
plt.close()

# Accuracy Graphs
plt.plot(train_accuracy_list, label = 'Training Accuracy')
plt.plot(val_accuracy_list, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy VS Epoch')
plt.legend()
plt.savefig('./graphs/sequence_prediction/sequence_prediction_trainingTestingAccuracy.png')
plt.close()

# save Model
torch.save(model, 'model/sequence_prediction.pt')
print('Graphs generated and saved')