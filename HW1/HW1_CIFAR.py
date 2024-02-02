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

# Network Parameters
imageSize = 32*32*3
hs1 = 150 # size of hidden layer 1
hs2 = 200 # size of hidden layer 2
hs3 = 100 # size of hidden layer 3
outputSize = 10 # 10 classes

# Training parameters
epochs = 20
learning_rate = 0.01

# Create a file for saving outputs
training_log_file = open('graphs/training_'+ str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'Epochs: {epochs}, Learning Rate: {learning_rate}, hs1: {hs1}, hs2: {hs2}, hs3: {hs3}' + '\n\n')
training_log_file.flush()

# Loading the data
# Define Transformation for downloading CIFAR-10
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

# Load the CIFAR-10 dataset
training_set = datasets.CIFAR10(root = './data', train = True, download=True, transform=transformation)
testing_set = datasets.CIFAR10(root = './data', train = False, download=True, transform=transformation)

# Training Set has 50,000 images (83%)
# Testing Set has 10,000 images (16.67%)

# Labels for CIFAR-10 (10 classes total)
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

# Setting up data loader
training_loader = DataLoader(dataset = training_set, batch_size = 100, shuffle = True)
testing_loader = DataLoader(dataset = training_set, batch_size = 100, shuffle = False)

# Defining the network
class ImageCifarNet(nn.Module):
    def __init__(self):
        super(ImageCifarNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(imageSize, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, outputSize)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) # flatten image
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
temp = torch.cuda.is_available()
print('Is cuda Available:')
print(temp)

# Select the model to be used and use the GPU if is available
model = ImageCifarNet().to(device)

# Training Tracking
train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []
val_precision_list = []
val_recall_list = []
val_f1_list = []

data_predictions = []
data_targets = []

total_time = 0

# Select loss function and the optimizer
criterion = nn.CrossEntropyLoss() # Use Cross Entropy for classification
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# Training loop
print(f'Starting training for {epochs} epochs')
for epoch in range(epochs):
    start_time = time.time() # Get start time of epoch
    model.train()
    data_predictions = []
    data_targets = []
    for inputs, targets in training_loader: # For each epoch
        inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        data_predictions.extend(predicted.tolist())
        data_targets.extend(targets.tolist())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    # Calculate training accuracy for the epoch and save it
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
        for inputs, targets in testing_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            data_predictions.extend(predicted.tolist())
            data_targets.extend(targets.tolist())
        end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    # Calculate validation accuracy, precision, recall, and F1 score
    accuracy = 100 * sum([p == t for p, t in zip(data_predictions, data_targets)]) / len(data_targets)
    precision = precision_score(data_targets, data_predictions, average='weighted')
    recall = recall_score(data_targets, data_predictions, average='weighted')
    f1 = f1_score(data_targets, data_predictions, average='weighted')

    val_accuracy_list.append(accuracy)
    val_precision_list.append(precision)
    val_recall_list.append(recall)
    val_f1_list.append(f1)

    print(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Accuracy: {accuracy}% || Precision: {precision} || Recall: {recall} || F1 Score: {f1}')
    # Save the validation data to the log file
    training_log_file.write(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Accuracy: {accuracy}% || Precision: {precision} || Recall: {recall} || F1 Score: {f1}' + '\n')
    training_log_file.flush()


print(f'Finished Training {epochs} epochs in {total_time} seconds')
# Save the data to the log file
training_log_file.write(f'Finished Training {epochs} epochs in {total_time} seconds' + '\n')
training_log_file.flush()

# Generate Graphs
# Training Loss Graph
plt.plot(train_loss_list, label = 'Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss VS Epoch')
plt.savefig('./graphs/trainingLoss.png')
plt.close()

# Accuracy Graphs
plt.plot(train_accuracy_list, label = 'Training Accuracy')
plt.plot(train_accuracy_list, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy VS Epoch')
plt.legend()
plt.savefig('./graphs/trainingTestingAccuracy.png')
plt.close()

# Validation Graphs
plt.plot(val_f1_list, label = 'F1')
plt.plot(val_precision_list, label = 'Precision')
plt.plot(val_recall_list, label = 'Recall')
plt.xlabel('Epoch')
plt.title('Validation Metrics VS Epoch')
plt.legend()
plt.savefig('./graphs/validationMetrics.png')
plt.close()

# Confusion Matrix
# Used this guide: 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
cm = confusion_matrix(data_targets, data_predictions)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels).plot()
plt.savefig('./graphs/confusionMatrix.png')
plt.close

# save Model
torch.save(model, 'model/HousingModel.pt')
print('Graphs generated and saved')


