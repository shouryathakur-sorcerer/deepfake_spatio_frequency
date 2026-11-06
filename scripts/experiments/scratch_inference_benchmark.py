import time
import torch
from models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = ['spatial_only', 'frequency_only', 'naive_concat', 'fft_baseline', 'parallel_se', 'cross_gated']

print(f"Inference Latency Benchmark (NVIDIA GeForce RTX 4060 Laptop GPU):")
header = f"{'Model':<18} | {'Params':>12} | {'Batch-1 (ms)':>14} | {'Batch-64 (ms)':>14} | {'Throughput (img/s)':>20}"
print(header)
print("-" * len(header))

for name in models:
    m = get_model(name).to(device)
    m.eval()
    
    # Warmup
    x1 = torch.randn(1, 3, 256, 256, device=device)
    x64 = torch.randn(64, 3, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = m(x1)
            _ = m(x64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Benchmark Batch-1 (pure latency)
    t0 = time.time()
    n_b1 = 50
    with torch.no_grad():
        for _ in range(n_b1):
            _ = m(x1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b1_ms = (time.time() - t0) / n_b1 * 1000
    
    # Benchmark Batch-64 (throughput)
    t0 = time.time()
    n_b64 = 20
    with torch.no_grad():
        for _ in range(n_b64):
            _ = m(x64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b64_ms = (time.time() - t0) / n_b64 * 1000
    throughput = (64 * n_b64) / (b64_ms * n_b64 / 1000)
    
    p_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:<18} | {p_count:>12,d} | {b1_ms:>14.2f} | {b64_ms:>14.2f} | {throughput:>20.1f}")
