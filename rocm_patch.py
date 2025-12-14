"""
ROCm compatibility patch for VoxCPM
Fixes various torch operations that have issues on ROCm devices
"""

import torch
import sys


def patch_tensor_operations():
    """
    Patch various tensor operations that may fail on ROCm.
    """
    # Patch nonzero operation
    original_nonzero = torch.Tensor.nonzero

    def patched_nonzero(self, *args, **kwargs):
        try:
            return original_nonzero(self, *args, **kwargs)
        except RuntimeError as e:
            if "hipErrorInvalidDeviceFunction" in str(e):
                # CPU fallback for nonzero operation
                if self.is_cuda:
                    device = self.device
                    cpu_tensor = self.cpu()
                    result = original_nonzero(cpu_tensor, *args, **kwargs)
                    if hasattr(result, 'to'):
                        return result.to(device)
                    return result
            raise

    torch.Tensor.nonzero = patched_nonzero

    # Patch eq operation for tensors
    original_eq = torch.Tensor.__eq__

    def patched_eq(self, other):
        try:
            return original_eq(self, other)
        except RuntimeError as e:
            if "hipErrorInvalidDeviceFunction" in str(e):
                # CPU fallback for eq operation
                if self.is_cuda:
                    device = self.device
                    cpu_self = self.cpu()
                    cpu_other = other.cpu() if hasattr(other, 'cpu') and other.is_cuda else other
                    result = original_eq(cpu_self, cpu_other)
                    if hasattr(result, 'to'):
                        return result.to(device)
                    return result
            raise

    torch.Tensor.__eq__ = patched_eq

    print("Applied tensor operations patch for ROCm")


def patch_tensor_mask():
    """
    Patch tensor masking operations that may fail on ROCm.
    """
    original_tensor_mask = torch.Tensor.__getitem__

    def masked_tensor_getitem(self, mask):
        try:
            return original_tensor_mask(self, mask)
        except RuntimeError as e:
            if "hipErrorInvalidDeviceFunction" in str(e):
                # CPU fallback for masking operations
                if self.is_cuda and isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                    device = self.device
                    mask_device = mask.device if mask.is_cuda else None

                    # Move both to CPU
                    cpu_self = self.cpu()
                    cpu_mask = mask.cpu() if mask.is_cuda else mask

                    # Apply mask on CPU
                    result = cpu_self[cpu_mask]

                    # Move result back to original device
                    return result.to(device)
            raise

    # Apply the patch
    torch.Tensor.__getitem__ = masked_tensor_getitem


def patch_unique_consecutive():
    """
    Patch torch.unique_consecutive to handle ROCm compatibility issues.
    This provides a fallback implementation using CPU when the GPU version fails.
    """
    original_unique_consecutive = torch.unique_consecutive

    def patched_unique_consecutive(input, *args, **kwargs):
        try:
            # Try the original implementation first
            return original_unique_consecutive(input, *args, **kwargs)
        except RuntimeError as e:
            if "hipErrorInvalidDeviceFunction" in str(e) or "unique failed" in str(e):
                # If ROCm error, fallback to CPU implementation
                if input.is_cuda:
                    # Move to CPU, process, then move back to original device
                    device = input.device
                    input_cpu = input.cpu()

                    # Apply unique_consecutive on CPU
                    result_cpu = original_unique_consecutive(input_cpu, *args, **kwargs)

                    # Move result back to GPU if it's a tensor
                    if isinstance(result_cpu, torch.Tensor):
                        return result_cpu.to(device)
                    elif isinstance(result_cpu, tuple):
                        # Handle tuple of tensors
                        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in result_cpu)

                # Re-raise if not ROCm-related or fallback failed
                raise
            else:
                # Re-raise if it's a different error
                raise

    # Replace the function
    torch.unique_consecutive = patched_unique_consecutive
    print("Applied ROCm compatibility patch for torch.unique_consecutive")


def patch_model_inference():
    """
    Patch funasr model inference to use CPU fallback for problematic operations.
    """
    # Force the ASR model to use CPU for certain operations if on ROCm
    import os
    if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
        # Set environment variables to help with ROCm compatibility
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
        # Disable some optimizations that may cause issues on ROCm
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
        print("Set ROCm optimization environment variables")


def apply_patches():
    """Apply all ROCm compatibility patches"""
    if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
        print("Detected ROCm, applying compatibility patches...")

        # Apply patches in order
        patch_model_inference()
        patch_unique_consecutive()

        # Patch tensor operations
        try:
            patch_tensor_operations()
            print("Applied tensor operations patch")
        except Exception as e:
            print(f"Failed to patch tensor operations: {e}")

        try:
            patch_tensor_mask()
            print("Applied tensor masking patch")
        except Exception as e:
            print(f"Failed to patch tensor masking: {e}")

        print("All ROCm compatibility patches applied")
    else:
        print("ROCm not detected, skipping patches")


# Apply patches when imported
if __name__ == "__main__":
    apply_patches()