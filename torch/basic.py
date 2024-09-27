import torch

from utils import GPU, simple_timer

def create_zero_gpu():
    n = 10000
    dev = GPU
    
    for _ in range(0, n):
        t = torch.zeros(1024, 8192, dtype=torch.float32, device=dev)

def create_ones_gpu():
    n = 10000
    dev = GPU
    
    for _ in range(0, n):
        t = torch.ones(1024, 8192, dtype=torch.float32, device=dev)


ctg = simple_timer(create_zero_gpu, GPU)
print(f"Zeroes: {ctg}s {ctg / 10000}s/iter")
del ctg

ctg = simple_timer(create_ones_gpu, GPU)
print(f"Ones: {ctg}s {ctg / 10000}s/iter")
del ctg