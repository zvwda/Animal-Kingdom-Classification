import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.class_mapping = {
            "beaver": 0, "butterfly": 1, "cougar": 2, "crab": 3, "crayfish": 4,
            "crocodile": 5, "dolphin": 6, "dragonfly": 7, "elephant": 8,
            "flamingo": 9, "kangaroo": 10, "leopard": 11, "llama": 12,
            "lobster": 13, "octopus": 14, "pigeon": 15, "rhino": 16, "scorpion": 17
        }
        
        print(f"Original dataset size: {len(self.df)}")
        print("Original class distribution:")
        print(self.df.iloc[:, 1].value_counts())
        
        self.class_counts = self.df.iloc[:, 1].value_counts()
        self.max_size = self.class_counts.max()
        
        balanced_data = []
        for class_name in self.class_mapping.keys():
            class_data = self.df[self.df.iloc[:, 1] == class_name]
            class_size = len(class_data)
            balanced_data.append(class_data)
            
            if class_size < self.max_size:
                num_augment = self.max_size - class_size
                augment_indices = np.random.choice(class_data.index, num_augment)
                augmented = self.df.loc[augment_indices].copy()
                balanced_data.append(augmented)
        
        self.df = pd.concat(balanced_data, ignore_index=True)
        print(f"\nAugmented dataset size: {len(self.df)}")
        print("Balanced class distribution:")
        print(self.df.iloc[:, 1].value_counts())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.image_folder, row.iloc[0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.class_mapping[row.iloc[1]]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.1, interpolation=2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]) 


csv_file = '/kaggle/input/animal-kingdom-classification/AnimalTrainData/AnimalTrainData/train.csv'
image_folder = '/kaggle/input/animal-kingdom-classification/AnimalTrainData/AnimalTrainData'


dataset = CustomDataset(csv_file=csv_file, image_folder=image_folder, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=18):
        super(CustomResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckResidualBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class BottleneckResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


from torchvision.models import ResNet50_Weights

pretrained_resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

pretrained_resnet.fc = nn.Linear(2048, 18)

pretrained_dict = pretrained_resnet.state_dict()

custom_model = CustomResNet50(num_classes=18)
custom_dict = custom_model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_dict}
custom_dict.update(pretrained_dict)
custom_model.load_state_dict(custom_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model = custom_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)  #learning rate will update after each 7 epochs with 0.5*lr bta3y

best_val_accuracy = 0
patience = 7
no_improve_epochs = 0

overfitting_patience = 5
overfit_epochs = 0

train_losses, val_accuracies, train_accuracies = [], [], []
num_epochs = 10

for epoch in range(num_epochs):
    custom_model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = custom_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    train_losses.append(running_loss / len(train_loader))

    custom_model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = custom_model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if epoch > 0 and train_accuracies[-1] - val_accuracies[-1] > 5:
        overfit_epochs += 1
        print("Warning: Possible overfitting detected!")
    else:
        overfit_epochs = 0

    if overfit_epochs >= overfitting_patience:
        print("Early stopping triggered due to overfitting!")
        break

    if no_improve_epochs >= patience:
        print("Early stopping triggered due to no improvement in validation accuracy!")
        break

    scheduler.step()

average_validation_accuracy = np.mean(val_accuracies)
print(f"Average Validation Accuracy: {average_validation_accuracy:.2f}%")