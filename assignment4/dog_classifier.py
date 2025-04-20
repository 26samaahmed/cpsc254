import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import time
from sklearn.metrics import accuracy_score


class DogDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {}

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                breed = filename.split('_')[0]
                if breed not in self.label_map:
                    self.label_map[breed] = len(self.label_map)
                label = self.label_map[breed]
                self.images.append(os.path.join(image_folder, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


dataset_dir = 'L05_DL_Vision_Dogs'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_data = DogDataset(dataset_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes = len(train_data.label_map)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


print("\n===== Model Architecture Info =====")
print(f"Total number of layers (modules): {len(list(model.modules()))}")
print(f"First conv layer filters: {model.conv1.out_channels}")
print(f"First conv kernel size: {model.conv1.kernel_size}")
print(f"Fully connected output features: {model.fc.out_features}")
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("====================================\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training model...")
model.train()
for epoch in range(15):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

print("\nEvaluating model...")
model.eval()
start_time = time.time()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

end_time = time.time()

accuracy = accuracy_score(all_labels, all_preds)
classification_time = end_time - start_time

print("\n===== Evaluation Results =====")
print(f"Classification accuracy: {accuracy * 100:.2f}%")
print(f"Total classification time: {classification_time:.2f} seconds")
print("==============================")