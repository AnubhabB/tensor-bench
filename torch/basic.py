import torch

from utils import GPU, simple_timer

N = 64

def create_zero_gpu():
    dev = GPU
    
    for _ in range(0, N):
        t = torch.zeros(1024, 8192, dtype=torch.float32, device=dev)

def create_ones_gpu():
    dev = GPU
    
    for _ in range(0, N):
        t = torch.ones(1024, 8192, dtype=torch.float32, device=dev)

def create_full_gpu():
    dev = GPU
    
    for i in range(0, N):
        t = torch.full((1024, 8192), i, dtype=torch.float32, device=dev)


ctg = simple_timer(create_zero_gpu, GPU)
print(f"Zeroes: {ctg}s {ctg / 10000}s/iter")

ctg = simple_timer(create_ones_gpu, GPU)
print(f"Ones: {ctg}s {ctg / 10000}s/iter")

ctg = simple_timer(create_full_gpu, GPU)
print(f"Full: {ctg}s {ctg / 10000}s/iter")