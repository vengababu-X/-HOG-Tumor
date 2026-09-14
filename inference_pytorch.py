import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

print("="*60)
print("     FREEBUFF V2.0 - PYTORCH INFERENCE ENGINE")
print("="*60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running inference on: {device}")

# 1. Define class labels
class_names = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

# 2. Rebuild architecture and load saved weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

weights_path = Path("saved_models/quick_gpu_test.pth")
if not weights_path.exists():
    weights_path = Path("saved_models/pytorch_gpu_resnet18.pth")

if weights_path.exists():
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"[SUCCESS] Loaded weights from: {weights_path}")
else:
    print(f"[ERROR] No weights found at {weights_path}. Please train the model first.")
    exit(1)

model = model.to(device)
model.eval()

# 3. Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Prediction function for a given image path
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = class_names[predicted_idx.item()]
        confidence = conf.item() * 100
        
        print(f"\n[RESULT] Image: {Path(image_path).name}")
        print(f" -> Predicted Class: {predicted_class}")
        print(f" -> Confidence: {confidence:.2f}%\n")
    except Exception as e:
        print(f"[ERROR] Could not process image: {e}")

# Test with a dummy image or prompt user
print("[INFO] Inference pipeline ready! You can now call predict_image('path_to_scan.jpg') in your code.")
