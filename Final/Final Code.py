import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import seaborn as sn
import pandas as pd
from math import ceil
from tqdm import tqdm
from DataLoader import dataset, Animals_train_loader, Animals_test_loader, Animals_map, Animals_map_flipped, birds_map
from modelv2_attempt2 import EfficientNet
from modelv2_resnet import ResNet18_96x96, ResNet_block, ResNet34_224x224
from PIL import Image
from efficientnet_pytorch import EfficientNet
#from efficientnet import EfficientNetB0 as Net








# Training parameters
epochs = 5
learning_rate = 0.0001 # 0.000005 # 0.0001 was best for pretrained efficientNet

# Create a file for saving outputs
training_log_file = open('graphs/training_' + str(time.time()) +'.log', 'w')

# Save parameters to log
training_log_file.write(f'Epochs: {epochs}, Learning Rate: {learning_rate}' + '\n\n')
training_log_file.flush()

# Loading the data
# Define Transformation for downloading CIFAR-10
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

# Load the CIFAR-10 dataset
#training_set = datasets.CIFAR10(root = './data', train = True, download=True, transform=transformation)
#testing_set = datasets.CIFAR10(root = './data', train = False, download=True, transform=transformation)

# Training Set has 50,000 images (83%)
# Testing Set has 10,000 images (16.67%)

# Labels for Animal data set
'''
class_labels = {'Abbotts Babbler Malacocincla abbotti', 'Black Bittern (Dupetor flavicollis)', 
             'Blue-eared Kingfisher Alcedo meninting', 'Blue-naped Pitta Pitta nipalensis', 
             'Broad-billed Warbler Tickellia hodgsoni', 'Cheer Pheasant (Catreus wallichii)', 
             'Chestnut Munia Lonchura atricapilla', 'Cinereous Vulture Aegypius monachus', 
             'Golden Babbler Stachyris chrysaea', 'Goulds Shortwing Brachypteryx stellata', 
             'Great Bittern Botaurus stellaris', 'Great Hornbill (Buceros bicornis)', 
             'Great Slaty Woodpecker Mulleripicus pulverulentus', 
             'Ibisbill Ibidorhyncha struthersii', 'Indian Courser Cursorius coromandelicus', 
             'Indian Grassbird - Graminicola bengalensis', 'Indian Nightjar Caprimulgus asiaticus', 
             'Knob-billed Duck Sarkidiornis melanotos', 'Northern Pintail Anas acuta', 
             'Painted Stork Mycteria leucocephala', 'Purple Cochoa Cochoa purpurea', 
             'Red-headed Trogon Harpactes erythrocephalus', 'Red-headed Vulture Sarcogyps calvus', 
             'Red-necked Falcon Falco chicquera', 'Ruby-cheeked Sunbird Anthreptes singalensis', 
             'Rusty-fronted Barwing Actinodura egertoni', 'Saker Falcon Falco cherrug', 
             'Silver-eared Mesia Leiothrix argentauris', 'Slaty-legged Crake Rallina eurizonoides', 
             'Spot-bellied Eagle Owl Bubo nipalensis', 'Sultan Tit Melanochlora sultanea', 
             'Swamp Francolin Francolinus gularis', 'Tawny-bellied Babbler Dumetia hyperythra', 
             'Thick-billed Green Pigeon Treron curvirostra', 'White-throated Bulbul Alophoixus flaveolus', 
             'White-throated Bushchat Saxicola insignis', 'Yellow-rumped Honeyguide - Indicator xanthonotus', 
             'Yellow-vented Warbler Phylloscopus cantator'}

'''
# Setting up data loader
class_labels = {"endangered", "predator", "prey"} #Animals_map_flipped



# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu") 
print(f'Using device: {device}')
temp = torch.cuda.is_available()
print('Is cuda Available:')
print(temp)

# Select the model to be used and use the GPU if is available
#model = ImageCifarNet().to(device)
#model = EfficientNet(SE_reduce=3,MB_reduce=3,MB_widening=2,outputClasses=90).to(device)
#model = EfficientNet(SE_reduce=3,outputClasses=2).to(device)
#model = ResNet18_32x32(ResNet_block).to(device)
#model = EfficientNet().to(device)

model_name = "ResNet"

if model_name == "EfficientNet":
    model = EfficientNet(SE_reduce=3,outputClasses=3).to(device)
elif model_name == "ResNet":
    model = ResNet34_224x224(ResNet_block).to(device)
elif model_name == "Pretrained ResNet":
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
#    for param in model.parameters():
#        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)
    model = model.to(device)  
elif model_name == "Pretrained EfficientNet":
    #model = models.efficientnet_b1(pretrained=True)
#    for param in model.parameters():
#        param.requires_grad = False
    #num_ftrs = model.classifier[1].in_features
    #model._fc = torch.nn.Linear(num_ftrs, 3)
    #model.classifier[1].out_features = 3
    #model = model.to(device)  
    
    model_name = 'efficientnet-b2'
    model = EfficientNet.from_pretrained(model_name, num_classes=3).to(device)


# Save how many parameters are in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters in the model: {total_params}')
training_log_file.write(f'Number of parameters in model: {total_params}' + '\n')
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
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 




# Training loop
print(f'Starting training for {epochs} epochs')
for epoch in range(epochs):
    print(f"Starting training for epoch {epoch+1}/{epochs}")
    start_time = time.time() # Get start time of epoch
    model.train()
    data_predictions = []
    data_targets = []
    for inputs, targets in tqdm(Animals_train_loader, total=len(Animals_train_loader)): # For each epoch
        inputs, targets = inputs.to(device), targets.to(device) # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        data_predictions.extend(predicted.tolist())
        data_targets.extend(targets.squeeze(1).tolist())
        loss = criterion(outputs, targets.squeeze(1))
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

    if (epoch + 1) == epochs:
        cm = confusion_matrix(data_targets, data_predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels).plot()
        plt.savefig('./graphs/Learning_confusionMatrix.png')
        plt.close()

    # Evaluation Loop
    print(f'Beginning validation for epoch {epoch+1}/{epochs}')
    start_time = time.time() # Get start time of epoch
    model.eval()

    data_predictions = []
    data_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(Animals_test_loader, total=len(Animals_test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            data_predictions.extend(predicted.tolist())
            data_targets.extend(targets.squeeze(1).tolist())
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
plt.plot(val_accuracy_list, label = 'Validation Accuracy')
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

if False:
    import glob
    import cv2

    imgs_path = "data/randomPictures"

    data = []
    for img_path in glob.glob(imgs_path + "/*"):
        data.append(img_path)

    plt.figure(figsize=(15,12))
    model.eval()
    for i, image_path in enumerate(data):
        plt.subplot(4,4,i+1)
        image = Image.open(image_path)
        plt.imshow(image)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (96,96))
        transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225))
                            ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float).to(device)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_labels = list(class_labels)
        plt.title(class_labels[predicted.item()])
        plt.axis('off')
    plt.savefig('./graphs/Sample.png')
    plt.close

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


