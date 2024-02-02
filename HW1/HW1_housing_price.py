from re import T
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import pandas as pd

# Settings
batchSize = 16
epochs = 40
learning_rate = 0.001

useHotEncoding = True
normalizeData = True

# Network Parameters
inputSize = 21 # 21 different Features
hs1 = 500 # size of hidden layer 1
hs2 = 1000 # size of hidden layer 2
hs3 = 200 # size of hidden layer 3
hs4 = 20 # size of hidden layer 4
outputSize = 1 # 1 Value

# Create a file for saving outputs
training_log_file = open('graphs/housing_price_training_'+ str(time.time()) +'.log', 'w')

print('Pre-processing Data')
# Read in data
df = pd.read_csv('./data/house-train.csv')

# Important Features in the data set
usefull_cols = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF'
                , 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces'
                ,'LotFrontage','WoodDeckSF','OpenPorchSF'
                ,'ExterQual','Neighborhood','MSZoning'
                ,'Alley','LotShape','LandContour','Condition1','HouseStyle',
                'MasVnrType','SaleCondition','SalePrice']

# Remove unneccesary columns
df = df[usefull_cols].copy()

# Remove NA from the data set
df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].mean())
df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean())

# Convert Text to numbers
le = LabelEncoder()
df['ExterQual'] = le.fit_transform(df['ExterQual'])
df['Neighborhood'] = le.fit_transform(df['Neighborhood'])
df['MSZoning'] = le.fit_transform(df['MSZoning'])
df['Alley'] = le.fit_transform(df['Alley'])
df['LotShape'] = le.fit_transform(df['LotShape'])
df['LandContour'] = le.fit_transform(df['LandContour'])
df['Condition1'] = le.fit_transform(df['Condition1'])
df['HouseStyle'] = le.fit_transform(df['HouseStyle'])
df['MasVnrType'] = le.fit_transform(df['MasVnrType'])
df['SaleCondition'] = le.fit_transform(df['SaleCondition'])

# One hot encoding
if useHotEncoding:
    inputSize = 80 # Increases to 80 different Features
    df = pd.get_dummies(df, columns = ['Neighborhood','MSZoning','Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition'])
    df = df.replace({True: 1, False: 0})

# Write to log with settings
training_log_file.write('Housing Price\n' + f'Batch Size: {batchSize} Epochs: {epochs} One-Hot Encoding: {useHotEncoding} Learning Rate: {learning_rate} Normalize Data: {normalizeData}' + '\n')
training_log_file.write(f'inputSize: {inputSize} hs1: {hs1} hs2: {hs2}, hs3: {hs3}, hs4: {hs4}' + '\n\n')
training_log_file.flush()

# Make sure all columns are numbers and handle any values that are NAN
df = df.apply(pd.to_numeric, errors = 'coerce').fillna(0)

# Break the data up into input data and target values
targets = df['SalePrice'].values
values = df.drop(['SalePrice'], axis = 1).values

# Define the train and test set
# Split data 80% Train 20% Test
train_values, test_values, train_targets, test_targets = train_test_split(values, targets, test_size = 0.2, random_state = 42) 


# Put the training and test sets into tensors
train_values = torch.tensor(train_values, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)

test_values = torch.tensor(test_values, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

if normalizeData:
    p = 1
    train_values = nn.functional.normalize(train_values, dim = 1, p = p)
    test_values = nn.functional.normalize(test_values, dim = 1, p = p)

# Create Tensor Data Sets and Data Loaders
train_dataset = TensorDataset(train_values, train_targets)
test_dataset = TensorDataset(test_values, test_targets)

train_loader = DataLoader(dataset= train_dataset, batch_size = batchSize, shuffle=True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batchSize, shuffle=False)

# Define Network

class HOUSINGNet(nn.Module):
    def __init__(self):
        super(HOUSINGNet, self).__init__()
        self.fc1 = nn.Linear(inputSize, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, hs4)
        self.fc5 = nn.Linear(hs4, outputSize)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)

        return x

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Select the model to be used and use the GPU if is available
model = HOUSINGNet().to(device)

# Calculate total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in the model: {total_params}')
training_log_file.write('Number of parameters in model: ' + str(total_params) + '\n\n')
training_log_file.flush()

# Training Tracking
train_loss_list = []
train_rmse_list = []

val_loss_list = []
val_rmse_list = []

total_time = 0

# Select loss function and the optimizer
criterion = nn.MSELoss() # Use MSE for classification
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Training loop
print(f'Starting training for {epochs} epochs')
start_time = time.time() # Get start time of epoch
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
        optimizer.zero_grad()  # Clear existing gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters
        train_loss += loss.item() * inputs.size(0)  # Accumulate the loss

    # Calculate training accuracy for the epoch and save it
    train_loss_list.append(train_loss) # Save epoch loss in list

    # Calculate training RMSE
    rmse = np.sqrt(train_loss / len(train_loader.dataset))
    train_rmse_list.append(rmse)

    # Output current progress of training 
    end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time # Get elapsed time of epoch
    print(f'Epoch {epoch+1} / {epochs}, Loss: {loss.item()}, Time: {elapsed_time} seconds, Training RMSE: {rmse}')
    # Save the output to the log file
    training_log_file.write(f'Epoch {epoch+1} / {epochs}, Loss: {loss.item()}, Time: {elapsed_time} seconds, Training RMSE: {rmse}' + '\n')
    training_log_file.flush()
    total_time += elapsed_time



    # Evaluation Loop
    print(f'Beginning validation for epoch {epoch+1}')
    start_time = time.time() # Get start time of epoch
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Compute loss
            test_loss += loss.item() * inputs.size(0)  # Accumulate the loss
        end_time = time.time() # Get end time of epoch
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    # Calculate validation RMSE
    rmse = np.sqrt(test_loss / len(test_loader.dataset))
    val_rmse_list.append(rmse)

    print(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. RMSE: {rmse}')
    # Save the validation data to the log file
    training_log_file.write(f'Validation for epoch {epoch+1}/{epochs} finished in {elapsed_time} seconds. RMSE: {rmse}' + '\n')
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
plt.savefig('./graphs/housing_trainingLoss.png')
plt.close()

# Accuracy Graphs
plt.plot(val_rmse_list, label = 'Validation RMSE')
plt.plot(train_rmse_list, label = 'Training RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training/Testing RMSE VS Epoch')
plt.legend()
plt.savefig('./graphs/housing_trainingTestingRMSE.png')
plt.close()

# save Model
torch.save(model, 'model/HousingModel.pt')
print('Graphs generated and saved')



