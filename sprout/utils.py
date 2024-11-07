import torch.distributed as dist
import argparse
import socket
import torch.distributed as dist

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

    
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        print(f"Found free port: {free_port}")


def print_trainable_params(model):
    total = 0
    trainable = 0
    for v in model.parameters():
        total += v.numel()
        if v.requires_grad:
            trainable += v.numel()
    
    rank0_print(f"Total : {total}   Trainable : {trainable}   ({(trainable/total)*100:.2f})%")

if __name__=="__main__":
    find_free_port()