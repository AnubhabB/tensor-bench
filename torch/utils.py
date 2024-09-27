import time

import torch

CPU = "cpu"
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def simple_timer(f, dev):
    backend = torch.mps if dev == "mps" else torch.cuda if dev == "cuda" else torch.cpu
    start = backend.Event(enable_timing=True)
    end = backend.Event(enable_timing=True)

    start.record()
    f()
    end.record()
    backend.synchronize()

    return start.elapsed_time(end) / 1000