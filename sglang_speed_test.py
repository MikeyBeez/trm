import sglang
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")

# Example: run a PyTorch-heavy operation via sglang utilities
# We'll manually create tensors on GPU to simulate a typical sglang workload

size = 4096
x = torch.rand(size, size, device=device)
y = torch.rand(size, size, device=device)

# Warm-up
for _ in range(5):
    z = torch.matmul(x, y)

# Timed benchmark
start = time.time()
for _ in range(10):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"Time for 10 matmuls: {end - start:.4f} s")
print(f"Average per run: {(end - start)/10:.4f} s")

