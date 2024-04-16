import time
import torch
import optimized_bitlinear as obl
import torch.nn.functional as F

dtype = torch.float16
run = 100


# Helper function to create a matrix with only -1, 0, 1
def create_special_matrix(size, device, dtype):
    return torch.randint(-1, 2, size, device=device, dtype=dtype)


bias = torch.randn(8192, device='cuda', dtype=dtype)

# Example for 3D (batch processing)
A_batch = torch.randn(256, 256, 2048, device='cuda', dtype=dtype)
B_batch = create_special_matrix((8192, 2048), device='cuda', dtype=dtype)

# Timing F.linear for batch
for _ in range(run):
    _ = F.linear(A_batch, B_batch, bias)
start_time = time.time()
for _ in range(run):
    C_batch_torch = F.linear(A_batch, B_batch, bias)
torch_batch_time = time.time() - start_time

# Timing custom method for batch
for _ in range(run):
    _ = obl.mat_mul(A_batch, B_batch, bias)
start_time = time.time()
for _ in range(run):
    C_batch_custom = obl.mat_mul(A_batch, B_batch, bias)
custom_batch_time = time.time() - start_time

# Print results for batch processing
print("Batch Custom Method Time: {:.6f}s".format(custom_batch_time))
print("Batch Custom Method Shape:", C_batch_custom.shape)
print("Batch PyTorch F.linear Time: {:.6f}s".format(torch_batch_time))
print("Batch F.linear Shape:", C_batch_torch.shape)
print("Batch Max difference:", (C_batch_custom - C_batch_torch).abs().max())
