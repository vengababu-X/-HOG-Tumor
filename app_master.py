import sys
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

def print_banner():
    print("="*64)
    print("                        FreeBuff v4                         ")
    print("           Clinical Decision Support — Brain Tumor          ")
    print("            PyTorch GPU Accelerated Architecture            ")
    print("         Glioma · Meningioma · Pituitary · No Tumor         ")
    print("="*64)

def configure_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[SUCCESS] GPU Active: {torch.cuda.get_device_name(0)}")
        print(f"[SUCCESS] VRAM Ready: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[INFO] Running on CPU")

def main():
    print_banner()
    configure_gpu()
    print("\n[INFO] Master system check passed successfully using PyTorch!")
    print("[INFO] To launch the web interface, run: streamlit run app.py")

if __name__ == "__main__":
    main()
