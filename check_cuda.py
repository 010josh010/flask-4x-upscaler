"""Quick check: is CUDA available to torch?"""
import torch

print("CUDA available:", torch.cuda.is_available())
