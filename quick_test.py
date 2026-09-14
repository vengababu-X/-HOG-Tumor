import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

print("="*60)
print("     FAST GPU SMOKE TEST & TRAINING")
print("="*60)

device = torch.device("cuda:0")
print(f"[SUCCESS] Using GPU: {torch.cuda.get_device_name(0)}")

# Generate synthetic image tensors directly in GPU memory to test lightning-fast training
print("[INFO] Generating dummy dataset tensors for instant testing...")
inputs = torch.randn(200, 3, 144, 144)
labels = torch.randint(0, 4, (200,))
dataset = TensorDataset(inputs, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load ResNet18
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("[INFO] Training model on GPU for 3 epochs...")
start = time.time()
for epoch in range(3):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/3 | Loss: {running_loss/len(loader):.4f}")

print(f"[SUCCESS] Training test completed in {time.time() - start:.2f} seconds!")

os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/quick_gpu_test.pth")
print("[SUCCESS] Model weights saved to saved_models/quick_gpu_test.pth")
