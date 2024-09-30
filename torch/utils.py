import time
import torch

CPU = "cpu"
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def simple_timer(f, dev):
    backend = torch.mps if dev == "mps" else torch.cuda if dev == "cuda" else torch.cpu
    start = time.time() if dev == "cpu" else backend.Event(enable_timing=True)
    end = None if dev == "cpu" else backend.Event(enable_timing=True)
    
    if dev != "cpu":
        start.record()
    f()
    t = None
    if dev != "cpu":
        backend.synchronize()
        end.record()
        t = start.elapsed_time(end) / 1000
    else:
        t = time.time() - start

    return t