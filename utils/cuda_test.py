import torch

def check_torch_cuda_cudnn():
    # PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch Version: {torch_version}")

    # CUDA availability and version
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device = torch.cuda.current_device()
        cuda_capability = torch.cuda.get_device_capability(cuda_device)
        cuda_version = torch.version.cuda
        print(f"CUDA Available: Yes (CUDA Version: {cuda_version}, CUDA Capability: {cuda_capability})")
    else:
        print("CUDA Available: No")

    # cuDNN version
    if cuda_available:
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN Version: {cudnn_version}")
    else:
        print("cuDNN Version: Not applicable (CUDA is not available)")

if __name__ == "__main__":
    check_torch_cuda_cudnn()