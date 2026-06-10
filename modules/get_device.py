import torch

def get_optimal_device():
    """
    Detects the best available hardware accelerator for PyTorch.
    Checks for NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS), Intel (XPU), 
    and falls back to CPU safely.
    """
    # 1. Check for NVIDIA (CUDA) or AMD (ROCm)
    # Note: PyTorch maps AMD ROCm backend under the `cuda` namespace as well.
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device_name} (Using CUDA/ROCm backend)")
        return torch.device('cuda')
    
    # 2. Check for Apple Silicon (M1/M2/M3 Mac GPUs)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("GPU Detected: Apple Silicon (Using MPS backend)")
        return torch.device('mps')
    
    # 3. Check for Intel GPUs
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("GPU Detected: Intel GPU (Using XPU backend)")
        return torch.device('xpu')
    
    # 4. Fallback to CPU
    else:
        print("No compatible GPU found by PyTorch. Falling back to CPU.")
        print("   If your server has a GPU, you likely have the CPU-only version of PyTorch installed.")
        return torch.device('cpu')