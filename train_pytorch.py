import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

print("="*60)
print("       FREEBUFF V2.0 - PYTORCH ACCELERATED TRAINING")
print("="*60)

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Data Transforms & Loaders
data_dir = Path("dataset")
if not data_dir.exists():
    print("[ERROR] 'dataset' folder not found! Please ensure your image classes are inside a 'dataset' folder.")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

class_names = train_dataset.classes
print(f"[INFO] Detected Classes: {class_names}")
print(f"[INFO] Total Training Images: {len(train_dataset)}")

# 3. Model Setup (Transfer Learning with ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Fast Training Loop (Targeting < 10 mins)
num_epochs = 5
print(f"[INFO] Starting training for {num_epochs} epochs...")

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

elapsed = time.time() - start_time
print(f"[SUCCESS] Training completed in {elapsed/60:.2f} minutes!")

# 5. Save Weights
os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/pytorch_tumor_resnet18.pth"
torch.save(model.state_dict(), save_path)
print(f"[SUCCESS] PyTorch model weights saved to {save_path}")
