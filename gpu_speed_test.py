import torch
import time

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")

# Create a large tensor
size = 4096
x = torch.rand(size, size, device=device)
y = torch.rand(size, size, device=device)

# Warm-up (important for accurate timing)
for _ in range(10):
    z = x @ y

# Timing matrix multiplication
start = time.time()
for _ in range(100):
    z = x @ y
torch.cuda.synchronize()  # make sure all kernels finish
end = time.time()

print(f"Time for 100 matmuls of {size}x{size}: {end - start:.4f} seconds")

