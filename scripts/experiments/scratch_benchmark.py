import time
import torch
from models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = ['spatial_only', 'frequency_only', 'naive_concat', 'fft_baseline', 'parallel_se', 'cross_gated']
x = torch.randn(32, 3, 256, 256, device=device)
target = torch.ones(32, 1, device=device)
criterion = torch.nn.BCEWithLogitsLoss()

print(f"Benchmarking training steps per model (batch_size=32 on {torch.cuda.get_device_name(0)}):")
header = f"{'Model':<18} | {'Params':>12} | {'Step Time (ms)':>15} | {'Peak VRAM (MB)':>15}"
print(header)
print("-" * len(header))

for name in models:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m = get_model(name).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    # Warmup
    for _ in range(3):
        opt.zero_grad()
        loss = criterion(m(x), target)
        loss.backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n_steps = 15
    for _ in range(n_steps):
        opt.zero_grad()
        loss = criterion(m(x), target)
        loss.backward()
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = (time.time() - t0) / n_steps * 1000
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    p_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:<18} | {p_count:>12,d} | {dt:>15.2f} | {peak_mem:>15.1f}")
