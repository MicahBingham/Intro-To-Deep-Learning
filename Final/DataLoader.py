from torchvision import datasets, transforms

import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# opening the file in read mode 
my_file = open("data/Animal Species/name of the animals.txt", "r") 

# reading the file 
data = my_file.read() 


# splitting the text it further when '\n' is seen. 
data_into_list = data.split("\n") 

# printing the data 
#print(data_into_list) 
my_file.close()

Animals_map = {}
for i, label in enumerate(data_into_list):
    Animals_map[label] = i

Animals_map_flipped = {}
for i, label in enumerate(data_into_list):
    Animals_map[i] = label

#print(class_map)

class AnimalsDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/Animal Species/animals/animals/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = Animals_map
        self.img_dim = (112,112) # 112 best for pretrained resnet
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
                          ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


prey = ['antelope', 'badger', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'cow', 'deer', 'donkey', 'dragonfly', 'duck', 'flamingo', 'fly', 'goat', 'goldfish', 'goose', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hornbill', 'horse', 'hummingbird', 'kangaroo', 'ladybugs', 'mosquito', 'mouse', 'oyster', 'parrot', 'penguin', 'pig', 'pigeon', 'porcupine', 'raccoon', 'rat', 'reindeer', 'sandpiper', 'seahorse', 'sheep', 'sparrow', 'squirrel', 'starfish', 'swan', 'turkey', 'wombat', 'woodpecker', 'zebra']
predator = ['bat', 'bear', 'boar', 'cat', 'coyote', 'crab', 'crow', 'dog', 'eagle', 'fox', 'hyena', 'jellyfish', 'lizard', 'lobster', 'moth', 'octopus', 'otter', 'owl', 'ox', 'seal', 'shark', 'snake', 'squid', 'turtle', 'wolf', 'pelecaniformes']
endangered = ['bison', 'chimpanzee', 'dolphin', 'elephant', 'gorilla', 'hippopotamus', 'koala', 'leopard', 'lion', 'okapi', 'orangutan', 'panda', 'pelecaniformes', 'possum', 'rhinoceros', 'tiger', 'whale']
endangered_preys = ['bison','chimpanzee', 'elephant','hippopotamus','koala', 'okapi', 'orangutan', 'panda', 'pelecaniformes', 'possum', 'rhinoceros']
endangered_predators = ['dolphin', 'leopard', 'lion','tiger', 'whale','gorilla']
birds_labels = ['Abbotts Babbler Malacocincla abbotti', 'Black Bittern (Dupetor flavicollis)', 
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
             'Yellow-vented Warbler Phylloscopus cantator']
endangered += birds_labels

class AnimalsDataset2(Dataset):
    def __init__(self):
        self.imgs_path = "data/Animal Species/animals/animals/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"endangered" : 0, "predator": 1, "prey": 2}
        self.img_dim = (64,64)  #(416, 416) # pretrained resnet 112x112
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        if class_name in prey:
            class_name = "prey"
        elif class_name in predator:
            class_name = "predator"
        elif class_name in endangered:
            class_name = "endangered"
        class_id = self.class_map[class_name]
        transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
                          ])
        #transform=transforms.Compose([transforms.ToTensor(),
        #                      transforms.Normalize((112.2546, 127.5798, 129.6500),
        #                                           (74.1950, 68.9958, 72.8105))
        #                  ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

class PPDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/Animal Species/animals/animals/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"predator": 0, "prey": 1}
        self.img_dim = (32,32)  #(416, 416)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        if (class_name in prey) or (class_name in endangered_preys):
            class_name = "prey"
        elif (class_name in predator) or (class_name in endangered_predators):
            class_name = "predator"
        class_id = self.class_map[class_name]

        #img = transforms.ToPILImage()(img)
        #if random.random() > 0.5:
        #    img = transforms.RandomHorizontalFlip()(img)
        #img = transforms.RandomRotation(15)(img)
        #img = transforms.RandomCrop(self.img_dim)(img)
        transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
                          ])
        #transform=transforms.Compose([transforms.ToTensor(),
        #                      transforms.Normalize((112.2546, 127.5798, 129.6500),
        #                                           (74.1950, 68.9958, 72.8105))
        #                  ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


class AnimalsBirdsDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/Animal Species/animals/animals/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                if (class_name in endangered) or (class_name in predator):
                    self.data.append([img_path, class_name])
                    #self.data.append([img_path, class_name])
                else:
                    self.data.append([img_path, class_name])
        #print(self.data)
        for type in ['train', 'val', 'test']:  #, 'test', 'val'
            self.imgs_path = f"data/Endangered Species/Dataset/{type}/"
            file_list = glob.glob(self.imgs_path + "*")
            for class_path in file_list:
                class_name = class_path.split("\\")[-1]
                for img_path in glob.glob(class_path + "/*"):
                    self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"endangered" : 0, "predator": 1, "prey": 2}
        self.img_dim = (32,32)  #(416, 416)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        if class_name in prey:
            class_name = "prey"
        elif class_name in predator:
            class_name = "predator"
        elif class_name in endangered:
            class_name = "endangered"
        class_id = self.class_map[class_name]
        transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
                          ])
        #transform=transforms.Compose([transforms.ToTensor(),
        #                      transforms.Normalize((112.2546, 127.5798, 129.6500),
        #                                           (74.1950, 68.9958, 72.8105))
        #                  ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

class EndangeredDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/Animal Species/animals/animals/"
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                if (class_name in endangered) or (class_name in predator):
                    self.data.append([img_path, class_name])
                    #self.data.append([img_path, class_name])
                else:
                    self.data.append([img_path, class_name])
        #print(self.data)
        for type in []:  #, 'test', 'val' # 'train', 'val', 'test'
            self.imgs_path = f"data/Endangered Species/Dataset/{type}/"
            file_list = glob.glob(self.imgs_path + "*")
            for class_path in file_list:
                class_name = class_path.split("\\")[-1]
                for img_path in glob.glob(class_path + "/*"):
                    self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"None" : 0, "endangered" : 1}
        self.img_dim = (64,64)  #(416, 416)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        if class_name in prey:
            class_name = "None"
        elif class_name in predator:
            class_name = "None"
        elif class_name in endangered:
            class_name = "endangered"
        class_id = self.class_map[class_name]
        transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
                          ])
        #transform=transforms.Compose([transforms.ToTensor(),
        #                      transforms.Normalize((112.2546, 127.5798, 129.6500),
        #                                           (74.1950, 68.9958, 72.8105))
        #                  ])
        img_normalized = transform(img)
        img_normalized = np.array(img_normalized)
        img_normalized = img_normalized.transpose(1, 2, 0)
        img_tensor = torch.from_numpy(img_normalized).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

birds_map = {'Abbotts Babbler Malacocincla abbotti': 0, 'Black Bittern (Dupetor flavicollis)': 1, 
             'Blue-eared Kingfisher Alcedo meninting': 2, 'Blue-naped Pitta Pitta nipalensis': 3, 
             'Broad-billed Warbler Tickellia hodgsoni': 4, 'Cheer Pheasant (Catreus wallichii)': 5, 
             'Chestnut Munia Lonchura atricapilla': 6, 'Cinereous Vulture Aegypius monachus': 7, 
             'Golden Babbler Stachyris chrysaea': 8, 'Goulds Shortwing Brachypteryx stellata': 9, 
             'Great Bittern Botaurus stellaris': 10, 'Great Hornbill (Buceros bicornis)': 11, 
             'Great Slaty Woodpecker Mulleripicus pulverulentus': 12, 
             'Ibisbill Ibidorhyncha struthersii': 13, 'Indian Courser Cursorius coromandelicus': 14, 
             'Indian Grassbird - Graminicola bengalensis': 15, 'Indian Nightjar Caprimulgus asiaticus': 16, 
             'Knob-billed Duck Sarkidiornis melanotos': 17, 'Northern Pintail Anas acuta': 18, 
             'Painted Stork Mycteria leucocephala': 19, 'Purple Cochoa Cochoa purpurea': 20, 
             'Red-headed Trogon Harpactes erythrocephalus': 21, 'Red-headed Vulture Sarcogyps calvus': 22, 
             'Red-necked Falcon Falco chicquera': 23, 'Ruby-cheeked Sunbird Anthreptes singalensis': 24, 
             'Rusty-fronted Barwing Actinodura egertoni': 25, 'Saker Falcon Falco cherrug': 26, 
             'Silver-eared Mesia Leiothrix argentauris': 27, 'Slaty-legged Crake Rallina eurizonoides': 28, 
             'Spot-bellied Eagle Owl Bubo nipalensis': 29, 'Sultan Tit Melanochlora sultanea': 30, 
             'Swamp Francolin Francolinus gularis': 31, 'Tawny-bellied Babbler Dumetia hyperythra': 32, 
             'Thick-billed Green Pigeon Treron curvirostra': 33, 'White-throated Bulbul Alophoixus flaveolus': 34, 
             'White-throated Bushchat Saxicola insignis': 35, 'Yellow-rumped Honeyguide - Indicator xanthonotus': 36, 
             'Yellow-vented Warbler Phylloscopus cantator': 37}


class BirdsDataset(Dataset):
    def __init__(self, type):
        self.imgs_path = f"data/Endangered Species/Dataset/{type}/" # data\Endangered Species\Dataset\train
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = birds_map
        self.img_dim = (64,64)  #(416, 416)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        assert os.path.exists(img_path)
        #print(img_path)
        open(img_path, "r")
        img = cv2.imread(img_path)
        #print(img)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img).to(torch.float)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id   


print("Loading Dataset")
batch_size = 512 #1024
dataset = AnimalsDataset2() # 3 classes

Animals_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

inputs = []
targets = []
for x, y in tqdm(Animals_data_loader,total=len(Animals_data_loader)):
    inputs.append(x)
    targets.append(y)
inputs = torch.cat(inputs, dim=0)
targets = torch.cat(targets, dim=0)
#inputs, targets = dataset
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, train_size=0.7, stratify=targets)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

Animals_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
Animals_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

if __name__ == "__main__":
    print('Dataloader is completed')
    #dataset = CustomDataset()
    #data_loader = DataLoader(dataset, batch_size=128, shuffle=True)


    #for imgs, labels in data_loader:
    #    print("Batch of images has shape: ", imgs.shape)
    #    print("Batch of labels has shape: ", labels.shape)
        