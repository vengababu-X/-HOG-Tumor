import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

print("="*60)
print("     FREEBUFF V2.0 - HARDWARE ENFORCED GPU TRAINING")
print("="*60)

# Force CUDA checking
assert torch.cuda.is_available(), "[FATAL ERROR] PyTorch cannot detect your NVIDIA GPU! Check your CUDA drivers."
device = torch.device("cuda:0")
torch.cuda.set_device(0)

print(f"[SUCCESS] GPU LOCKED: {torch.cuda.get_device_name(0)}")
print(f"[INFO] VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Point to the dataset directory
data_dir = Path(r"C:\Users\Bolig\OneDrive\Desktop\team project  4 year\dataset")
if not data_dir.exists():
    # Fallback search in root
    data_dir = Path("dataset")

if not data_dir.exists():
    print(f"[ERROR] Dataset directory not found at: {data_dir}. Please make sure your dataset folder is in the root directory.")
    exit(1)

print(f"[INFO] Using Dataset Path: {data_dir}")

transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

class_names = train_dataset.classes
print(f"[INFO] Classes detected: {class_names}")
print(f"[INFO] Total images to process: {len(train_dataset)}")

# Load ResNet18 model onto GPU
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print(f"[INFO] Starting lightning-fast GPU training for {num_epochs} epochs...")

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
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
    print(f"Epoch [{epoch+1}/{num_epochs}] | GPU Loss: {epoch_loss:.4f} | GPU Accuracy: {epoch_acc*100:.2f}%")

elapsed = time.time() - start_time
print(f"[SUCCESS] GPU Training completed in {elapsed/60:.2f} minutes!")

os.makedirs("saved_models", exist_ok=True)
save_path = "saved_models/pytorch_gpu_resnet18.pth"
torch.save(model.state_dict(), save_path)
print(f"[SUCCESS] Model weights saved to {save_path}")
