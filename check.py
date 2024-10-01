import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print CUDA availability
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Print the name of each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA-enabled GPU found.")
