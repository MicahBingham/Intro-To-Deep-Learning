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
outputSize = 10 # 10 classes
useDropout = True
dropout = 0.5

# Training parameters
epochs = 20
learning_rate = 0.01
batchSize = 100

# Create a file for saving outputs
training_log_file = open('graphs/AlexNet_32x32/AlexNet_training_'+ str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'AlexNet 32x32' + '\n')
training_log_file.flush()
training_log_file.write(f'Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batchSize}, Using Dropout: {useDropout}, Dropout: {dropout*100}%' + '\n')
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
training_loader = DataLoader(dataset = training_set, batch_size = batchSize, shuffle = True)
testing_loader = DataLoader(dataset = training_set, batch_size = batchSize, shuffle = False)

# Defining the network
class AlexNet_32x32(nn.Module):
    def __init__(self):
        super(AlexNet_32x32, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=96,kernel_size = (11,11), stride = 2) # changed stride to 2 to account for size
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1) # changed stride to 1 to account for size

        self.conv2 = nn.Conv2d(in_channels = 96, out_channels=256,kernel_size=(5,5), stride=1,padding=2)

        self.conv3 = nn.Conv2d(in_channels= 256, out_channels=384, kernel_size=(3,3), stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels= 384, out_channels=384, kernel_size=(3,3), stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels= 384, out_channels=384, kernel_size=(3,3), stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=dropout)

        self.flat = nn.Flatten()
        self.fc6 = nn.Linear(6144,1000)
        self.fc7 = nn.Linear(1000,500)
        self.fc8 = nn.Linear(500,outputSize)
        

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))

        x = self.pool1(torch.relu(self.conv2(x)))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(torch.relu(self.conv5(x)))
        if useDropout:
            x = self.drop(x)

        #x = x.view(-1, 384*1*1) # Flatten
        x = self.flat(x)
        x = torch.relu(self.fc6(x))
        if useDropout:
            x = self.drop(x)
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        return x

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Select the model to be used and use the GPU if is available
model = AlexNet_32x32().to(device)

# Save Number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in the model: {total_params}')
training_log_file.write('Number of parameters in model: ' + str(total_params) + '\n\n')
training_log_file.flush()

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
            loss = criterion(outputs, targets)
        end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    val_loss_list.append(loss.item()) # Save epoch loss in list

    # Calculate validation accuracy, precision, recall, and F1 score
    accuracy = 100 * sum([p == t for p, t in zip(data_predictions, data_targets)]) / len(data_targets)
    precision = precision_score(data_targets, data_predictions, average='weighted')
    recall = recall_score(data_targets, data_predictions, average='weighted')
    f1 = f1_score(data_targets, data_predictions, average='weighted')

    val_accuracy_list.append(accuracy)
    val_precision_list.append(precision)
    val_recall_list.append(recall)
    val_f1_list.append(f1)

    print(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Loss: {loss.item()} || Accuracy: {accuracy}% || Precision: {precision} || Recall: {recall} || F1 Score: {f1}')
    # Save the validation data to the log file
    training_log_file.write(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. Loss: {loss.item()} || Accuracy: {accuracy}% || Precision: {precision} || Recall: {recall} || F1 Score: {f1}' + '\n')
    training_log_file.flush()


print(f'Finished Training {epochs} epochs in {total_time} seconds')
# Save the data to the log file
training_log_file.write(f'Finished Training {epochs} epochs in {total_time} seconds' + '\n')
training_log_file.flush()

# Generate Graphs
# Training Loss Graph
plt.plot(train_loss_list, label = 'Training Loss')
plt.plot(val_loss_list, label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss VS Epoch')
plt.legend()
plt.savefig('./graphs/AlexNet_32x32/AlexNet_32x32_trainingLoss.png')
plt.close()

# Accuracy Graphs
plt.plot(train_accuracy_list, label = 'Training Accuracy')
plt.plot(val_accuracy_list, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy VS Epoch')
plt.legend()
plt.savefig('./graphs/AlexNet_32x32/AlexNet_32x32_trainingTestingAccuracy.png')
plt.close()

# Validation Graphs
plt.plot(val_f1_list, label = 'F1')
plt.plot(val_precision_list, label = 'Precision')
plt.plot(val_recall_list, label = 'Recall')
plt.xlabel('Epoch')
plt.title('Validation Metrics VS Epoch')
plt.legend()
plt.savefig('./graphs/AlexNet_32x32/AlexNet_32x32_validationMetrics.png')
plt.close()

# Confusion Matrix
# Used this guide: 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
cm = confusion_matrix(data_targets, data_predictions)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels).plot()
plt.savefig('./graphs/AlexNet_32x32/AlexNet_32x32_confusionMatrix.png')
plt.close

# save Model
torch.save(model, 'model/AlexNet_32x32.pt')
print('Graphs generated and saved')