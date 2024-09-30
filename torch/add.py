import torch

from utils import GPU, simple_timer

N = 100

a_h = torch.rand(32,630,12,32, device="cpu")
b_h = torch.rand(32, 1, 1, 32, device="cpu")

a_d = torch.rand(32,630,12,32, device=GPU)
b_d = torch.rand(32,630,12,32, device=GPU)

def add_cpu():
    for _ in range(0, N):
        t = a_h + b_h
        # t = torch.full((1024, 8192), i, dtype=torch.float32, device=dev)

def add_gpu():
    for _ in range(0, N):
        t = a_d + b_d

ctg = simple_timer(add_cpu, "cpu")
print(f"Add_Cpu: {ctg}s {ctg / N}s/iter")

ctg = simple_timer(add_gpu, GPU)
print(f"Add_gpu: {ctg}s {ctg / N}s/iter")