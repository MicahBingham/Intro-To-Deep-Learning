import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
import requests

# Network Parameters
hiddenSize = 100
maxSequenceLength = 50
# Select which to use during training
useLSTM = False
useGRU = True

# Training parameters
epochs = 10
learning_rate = 0.01
batchSize = 64

# Create a file for saving outputs
training_log_file = open('graphs/shakespeare_prediction/shakespeare_prediction_'+ str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'Shakespeare_prediction' + '\n')
training_log_file.flush()
training_log_file.write(f'Using LSTM: {useLSTM}, Using GRU: {useGRU}' + '\n')
training_log_file.flush()

print("Loading Sequence...")

# Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
data = response.text  # This is the entire text data

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
training_log_file.write(f'Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batchSize}, Input Size: {inputSize}, Hidden Size: {hiddenSize}, Output Size: {outputSize}, Max Sequence Size: {maxSequenceLength}' + '\n')
training_log_file.flush()

class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=inputSize, embedding_dim=hiddenSize)
        
        self.lstm = nn.LSTM(input_size=hiddenSize,hidden_size=hiddenSize,batch_first=True)
        self.gru = nn.GRU(input_size=hiddenSize,hidden_size=hiddenSize,batch_first=True)

        self.fc = nn.Linear(in_features=hiddenSize,out_features=outputSize)
    
    def forward(self, x):
        x = self.embedding(x)

        # get output from selected network
        if useLSTM:
            out, _ = self.lstm(x)
        elif useGRU:
            out, _ = self.gru(x)
        # output the last time step for each sequence in the batch
        out = self.fc(out[:,-1,:])
        return out
    
# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# Step 3: Create a dataset class
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

train_dataset = CharDataset(X_train, y_train)
val_dataset = CharDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batchSize)
val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=batchSize)

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
    data_predictions = []
    data_targets = []
    model.train()
    for inputs, targets in train_loader: # For each epoch
        inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        # Add predictions to lists
        _, predicted = torch.max(outputs.data, 1)
        data_predictions.extend(predicted.tolist())
        data_targets.extend(targets.tolist())

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    accuracy = 100 * sum([p == t for p, t in zip(data_predictions, data_targets)]) / len(data_targets)
    train_accuracy_list.append(accuracy)
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
    data_predictions = []
    data_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader: # For each epoch
            inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            # Add predictions to lists
            _, predicted = torch.max(outputs.data, 1)
            data_predictions.extend(predicted.tolist())
            data_targets.extend(targets.tolist())

            loss = criterion(outputs, targets)

        end_time = time.time() # Get end time of epoch

    accuracy = 100 * sum([p == t for p, t in zip(data_predictions, data_targets)]) / len(data_targets)

    # Calculate training accuracy for the epoch and save it
    val_accuracy_list.append(accuracy)
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
testText = 'We are accounted poor citizens, the patricians good. What authority surfeits '
print('Test model:\n')
print(f'Input Text: {testText}\n')
print('Expected character: o\n')
initial_input = torch.tensor([char_to_ix[c] for c in testText[-maxSequenceLength:]], dtype=torch.long).unsqueeze(0).to(device)
predicted_char = model(initial_input)
predicted_index = torch.argmax(predicted_char, dim=1).item()
print(f'Predicted character: {ix_to_char[predicted_index]}\n')

# Log the Test
training_log_file.write(f'\nTesting model:\n')
training_log_file.flush()
training_log_file.write(f'Input Text: {testText}\n')
training_log_file.flush()
training_log_file.write('Expected character: o\n')
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
plt.savefig('./graphs/shakespeare_prediction/shakespeare_prediction_trainingLoss.png')
plt.close()

# Accuracy Graphs
plt.plot(train_accuracy_list, label = 'Training Accuracy')
plt.plot(val_accuracy_list, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy VS Epoch')
plt.legend()
plt.savefig('./graphs/shakespeare_prediction/shakespeare_prediction_trainingTestingAccuracy.png')
plt.close()

# save Model
torch.save(model, 'model/shakespeare_prediction.pt')
print('Graphs generated and saved')