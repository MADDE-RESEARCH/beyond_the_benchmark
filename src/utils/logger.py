import subprocess
import torch

def get_gpu_utilization():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return float(result.strip())
    except Exception:
        return None

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def estimate_tflops_utilization(model, batch_size, time_per_batch, peak_tflops=8.1): # Nvidia T4 GPU has Peak FP32 TFlops of 8.1
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_sample = 2 * trainable_params
    total_flops = flops_per_sample * batch_size
    tflops = total_flops / (time_per_batch * 1e12)
    utilization = tflops / peak_tflops
    return tflops, utilization, trainable_params