import gc
import torch
import collections

# For GPU memory helpers, refer to https://github.com/BlackHC/toma

def is_on_gpu(model):
    """
    Returns True if all parameters of a model live on the GPU.
    """
    assert isinstance(model, torch.nn.Module)
    on_gpu = True
    has_params = False
    for param in model.parameters():
        has_params = True
        if not param.data.is_cuda:
            on_gpu = False
    return has_params and on_gpu


def recursive_copy_to_device(value, device, non_blocking=True):
    """
    Recursively searches lists, tuples, dicts and copies any object which
    supports an object.to API (e.g. tensors) to device if possible.
    Other values are passed as-is in the result.

    Note:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """
    if isinstance(value, list) or isinstance(value, tuple):
        device_val = []
        for val in value:
            device_val.append(
                recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            )

        return device_val if isinstance(value, list) else tuple(device_val)
    elif isinstance(value, collections.abc.Mapping):
        device_val = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )

        return device_val
    elif callable(getattr(value, "to", None)):
        return value.to(device=device, non_blocking=non_blocking)

    return value


def gc_cuda():
    """Gargage collect Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_total_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0


def get_cuda_assumed_available_memory():
    if torch.cuda.is_available():
        return get_cuda_total_memory() - torch.cuda.memory_reserved()
    return 0


def get_cuda_available_memory():
    # Always allow for 1 GB overhead.
    if torch.cuda.is_available():
        return get_cuda_assumed_available_memory() - get_cuda_blocked_memory()
    return 0


def get_cuda_blocked_memory():
    if not torch.cuda.is_available():
        return 0

    available_memory = get_cuda_assumed_available_memory()
    current_block = available_memory - 2 ** 28  # 256 MB steps
    while True:
        try:
            block = torch.empty((current_block,), dtype=torch.uint8, device="cuda")
            break
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                current_block -= 2 ** 30
                if current_block <= 0:
                    return available_memory
            else:
                raise
    block = None
    gc_cuda()
    return available_memory - current_block


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def cuda_meminfo():
    if not torch.cuda.is_available():
        return

    print(
        "Total:", torch.cuda.memory_allocated() / 2 ** 30, " GB Cached: ", torch.cuda.memory_reserved() / 2 ** 30, "GB"
    )
    print(
        "Max Total:",
        torch.cuda.max_memory_allocated() / 2 ** 30,
        " GB Max Cached: ",
        torch.cuda.max_memory_reserved() / 2 ** 30,
        "GB",
    )
