from inference_pytorch import predict_image
import os

print("="*50)
print("     INTERACTIVE MRI BRAIN TUMOR PREDICTOR")
print("="*50)
while True:
    img_path = input("Enter the full path to your MRI image (or type 'exit' to quit): ").strip()
    if img_path.lower() == 'exit':
        break
    # Remove quotes if user dragged and dropped the file into terminal
    img_path = img_path.strip('"').strip("'")
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print(f"[ERROR] File not found: {img_path}. Please check the path and try again.\n")
